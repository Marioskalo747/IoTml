#Libraries
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from config import RANDOM_STATE

#pytorch neural network classifier compatible with sklearn API
class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    
    '''stores valus (contracts for working sklearn clone and get_params)'''
    def __init__(self, hidden=(256, 128, 64), dropout=0.2, lr=1e-3,
                 weight_decay=1e-4, batch_size=1024, max_epochs=60,
                 patience=6, class_weight=None, device=None,
                 random_state=42, verbose=False):
        self.hidden = hidden  #width of hidden layers
        self.dropout = dropout
        self.lr = lr #regularization
        self.weight_decay = weight_decay #L2 regularization
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience #epochs to wait for early stopping
        self.class_weight = class_weight #balanced or None
        self.device = device #pick gpu if available, else cpu
        self.random_state = random_state
        self.verbose = verbose
    
    
    '''builds blocks of linear, batchnorm, relu, dropout layers and final linear layer'''    
    def build(self, n_in, n_out):
        layers =  []
        prev = n_in
        for h in self.hidden:
            #batchnorm stabilizes training on features with different scales
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out)) #No softmax (crossentropyloss expects raw logits)
        return nn.Sequential(*layers)
    
    '''device choice'''
    def pick_device(self):
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    '''prepare data, build model, train with early stopping and class weights'''
    def fit(self, X, y):
        #reproducibility
        torch.manual_seed(self.random_state) 
        np.random.seed(self.random_state)
        #classes names to integers
        self.le_ = LabelEncoder().fit(y) 
        self.classes_ = self.le_.classes_
        yi = self.le_.transform(y)
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        d = X.shape[1]
        k = len(self.classes_)
        #10% validation split for early stopping
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        n_val = max(1, int(0.1*n))
        vi = idx[:n_val]
        ti = idx[n_val:]
        device= self.pick_device()
        try:
            self.fit_on(X, yi, ti, vi, d, k, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache() #gpu run out of memory, try cpu
            self.fit_on(X, yi, ti, vi, d, k, "cpu") 
        return self
    
    '''training loop with early stopping and class weights'''
    def fit_on(self, X, yi, ti, vi, d, k, device):
        model = self.build(d, k).to(device)
        if self.class_weight == "balanced": #weight per class N/(k*class_count)
            counts = np.bincount(yi, minlength=k).astype(np.float64)
            w = len(yi) / (k * counts.clip(min=1))
            weight = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            weight= None
        crit = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt = torch.tensor(X[ti]) #train data on cpu move per batch
        yt = torch.tensor(yi[ti], dtype=torch.long)
        Xv = torch.tensor(X[vi]).to(device) #validation 
        yv = torch.tensor(yi[vi], dtype= torch.long).to(device)
        #drop last batch if smaller than batch_size to avoid batchnorm issues
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True, drop_last=len(ti) > self.batch_size)
        best_val = np.inf
        best_state = None
        bad = 0 #consecutive epochs without improvement
        self.history_ = [] #learning curves
        for epoch in range(self.max_epochs):
            model.train() #enables dropout
            tot=0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad() #clear gradients
                loss = crit(model(xb), yb) 
                loss.backward() #backpropagation
                opt.step() #weights update
                tot += loss.item()*len(xb) #weighted by batch
            model.eval() #disables dropout
            with torch.no_grad():
                val = float(crit(model(Xv), yv))
            self.history_.append({"epoch":epoch, "train_loss": tot/len(ti), "val_loss": val})
            if self.verbose:
                print(f"epoch{epoch:3d} train{tot/len(ti):4f} val {val:.4f}")
            if val < best_val - 1e-5: #only a real improvement counts
                best_val = val
                bad = 0    
                best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()} #cpu copy of weights
            else:
                bad += 1
                if bad >= self.patience:
                    break #early stopping
        if best_state is not None:
            model.load_state_dict(best_state) #rollback
        model.eval()
        self.model_ = model.cpu() 
        self.n_features_in_ = d #sklearn comvention
        
    '''Compute logits in chunks (8192 rows) to avoid memory issues'''    
    def logits(self, X):
        x = np.asarray(X, dtype=np.float32)
        device = self.pick_device()
        model = self.model_.to(device) 
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), 8192):
                xb = torch.tensor(x[i:i + 8192]).to(device)
                outs.append(model(xb).cpu().numpy())
        self.model_ = model.cpu() #back to cpu to avoid gpu memory issues
        return np.vstack(outs)
    
    '''class prediction (returns class names)'''
    def predict(self, X):
        return self.le_.inverse_transform(self.logits(X).argmax(axis=1))
    
    '''probability prediction (softmax with per row subtraction for numerical stability)'''
    def predict_proba(self, X):
        z = self.logits(X)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)