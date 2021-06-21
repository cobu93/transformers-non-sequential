from sklearn import base, cluster, tree, metrics
import torch
import torch.nn as nn
import skorch
import os
import shutil
import numpy as np
import joblib

from sklearn import pipeline, preprocessing

from copy import deepcopy
## Callbacks

class RestoreCheckpoint(skorch.callbacks.Callback):
    def __init__(
        self, 
        f_params='params.pt', 
        f_optimizer='optimizer.pt', 
        f_criterion='criterion.pt', 
        f_history='history.json', 
        dirname=''
    ):

        self._f_params = f_params
        self._f_optimizer = f_optimizer
        self._f_criterion = f_criterion
        self._f_history = f_history
        self._dirname = dirname

       
    def get_params(self, deep=True):
        return {
            'f_params': self._f_params,
            'f_optimizer': self._f_optimizer,
            'f_criterion': self._f_criterion,
            'f_history': self._f_history,
            'dirname': self._dirname,
        }

    def on_train_end(self, net, **kwargs):
        print('Restoring checkpoint')

        net.load_params(
            f_params=os.path.join(self._dirname, self._f_params), 
            f_optimizer=os.path.join(self._dirname, self._f_optimizer), 
            f_criterion=os.path.join(self._dirname, self._f_criterion), 
            f_history=os.path.join(self._dirname, self._f_history)
            )
        
        print('Checkpoint restored')


class RestoreMoveCheckpoint(skorch.callbacks.Callback):
    def __init__(
        self, 
        monitor,
        f_params='params.pt', 
        f_optimizer='optimizer.pt', 
        f_criterion='criterion.pt', 
        f_history='history.json', 
        dirname=''
    ):

        self._monitor = monitor
        self._f_params = f_params
        self._f_optimizer = f_optimizer
        self._f_criterion = f_criterion
        self._f_history = f_history
        self._dirname = dirname

       
    def get_params(self, deep=True):
        return {
            'monitor': self._monitor,
            'f_params': self._f_params,
            'f_optimizer': self._f_optimizer,
            'f_criterion': self._f_criterion,
            'f_history': self._f_history,
            'dirname': self._dirname,
        }

    def on_train_end(self, net, **kwargs):
        print('Moving and restoring checkpoint')   

        # Pre moving
        for inv_epoch in reversed(net.history):
            if inv_epoch[self._monitor]:
                best_metric = inv_epoch[self._monitor.replace('_best', '')]
                break   

        # Restoring
        net.load_params(
            f_params=os.path.join(self._dirname, self._f_params), 
            f_optimizer=os.path.join(self._dirname, self._f_optimizer), 
            f_criterion=os.path.join(self._dirname, self._f_criterion), 
            f_history=os.path.join(self._dirname, self._f_history)
            )  

        # Moving
        count = 1
        checkpoint_path = os.path.join(self._dirname, '{}_{}'.format(best_metric, count))

        while os.path.exists(checkpoint_path):
            count += 1
            checkpoint_path = os.path.join(self._dirname, '{}_{}'.format(best_metric, count))

        os.mkdir(checkpoint_path)

        file_names = os.listdir(self._dirname)

        for file_name in file_names:
            src = os.path.join(self._dirname, file_name)
            if os.path.isfile(src):
                shutil.move(src, checkpoint_path)

        joblib.dump(net.get_params(), os.path.join(checkpoint_path, 'model_params.jl'))
        
        print('Checkpoint moved and restored to {}'.format(checkpoint_path))


## Transformers

class ReorderTransformer(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__( self, categorical_columns, numerical_columns):
        super().__init__()
        self._categorical_columns = categorical_columns
        self._numerical_columns = numerical_columns
        
    def get_params(self, deep=True):
        return {
            'categorical_columns': self._categorical_columns, 
            'numerical_columns': self._numerical_columns
        }
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        X = X[self._categorical_columns + self._numerical_columns]
        return X

class DTypeTransformer(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__(self, dtype):
        super().__init__()
        self._dtype = dtype
        
    def get_params(self, deep=True):
        return {
            'dtype': self._dtype
        }
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        return X.astype(self._dtype)

class LabelingTransformer(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__(self):
        super().__init__()
        self.n_features = 0
        self.encoders = []
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):
        self.n_features = X.shape[1]

        for i in range(self.n_features):
            ft_encoder = preprocessing.LabelEncoder()
            ft_encoder.fit(X.iloc[:, i])

            self.encoders.append(ft_encoder)

        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        for i in range(self.n_features):
            X.iloc[:, i] = self.encoders[i].transform(X.iloc[:, i])
        return X


class AttentionTransformer(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__(self, transformer_pipeline_file):
        super().__init__()
        self.transformer_pipeline_file = transformer_pipeline_file
        self.transformer_pipeline = joblib.load(transformer_pipeline_file)

        #self.preprocessing = self.transformer_pipeline['preprocessing']
        #self.classifier = self.transformer_pipeline['classifier']

        self.attn_weights = []
    
    def get_params(self, deep=True):
        return {
            'transformer_pipeline_file': self.transformer_pipeline_file,
        }
        
    def attention_extraction(self, s, input, output):
        self.attn_weights.append(output[1].detach().cpu().numpy())

    def fit(self, X, y=None):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):

        handlers = []

        for enc_layer in self.transformer_pipeline['classifier'].module_.transformer_encoder.layers:
            handlers.append(enc_layer.self_attn.register_forward_hook(self.attention_extraction))

        
        self.attn_weights = []
        _ = self.transformer_pipeline.predict(X)
        self.attn_weights = np.vstack(self.attn_weights).reshape(X.shape[0], -1)

        for handler in handlers:
            handler.remove()

        return self.attn_weights


class MultifeatureLabelingTransformer(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__(self):
        super().__init__()
        self.n_features = 0
        self.n_labels = 0
        self.encoders = []
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):
        self.n_features = X.shape[1]
        self.n_labels = 0

        for i in range(self.n_features):
            self.encoders.append({})
            uniques = X.iloc[:, i].unique()
            
            for unique in uniques:
                self.encoders[i][unique] = self.n_labels
                self.n_labels += 1
            
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):

        for i in range(self.n_features):
            X.iloc[:, i] = X.iloc[:, i].replace(self.encoders[i])
        
        return X

class DimensionTransformer(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__(self, dims):
        super().__init__()
        self._dims = dims
        
    def get_params(self, deep=True):
        return {
            'dims': self._dims
        }
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        x_exp = np.expand_dims(X, axis=self._dims)
        return x_exp

## Models

class FFModel(nn.Module):

    def __init__(self, n_inputs, hidden_sizes, n_outputs, dropouts):
        super(FFModel, self).__init__()
                
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        last_size = n_inputs
        
        for hidden_size, dropout in zip(hidden_sizes, dropouts):
            self.linears.append(nn.Linear(last_size, hidden_size))
            
            self.activations.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))
            last_size = hidden_size
        
        self.output = nn.Linear(last_size, n_outputs)
        
    def forward(self, inp):

        out = inp
        for lin, drop, act in zip(self.linears, self.dropouts, self.activations):
            out = lin(out)
            out = drop(out)
            out = act(out)
        
        return self.output(out)

class TransformerModelv3(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, n_num_cols, n_cat_cols, n_cat_labels, n_features, dropout=0.5):
        super(TransformerModelv3, self).__init__()
        
        self.n_num_cols = n_num_cols
        self.n_cat_cols = n_cat_cols

        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        self.cat_embedding = nn.Embedding(n_cat_labels, ninp)
        self.num_embedding = nn.ModuleList()

        for feature in range(n_num_cols):
            self.num_embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        
        self.decoder = nn.Linear(ninp * n_features, 1)
        
        self.activation = nn.Sigmoid()
        
    
    def forward(self, src):

        batch_size = src.shape[0]
        src_cat = self.cat_embedding(src[:, :self.n_cat_cols].squeeze().long())
        
        if batch_size == 1:
            src_cat = src_cat.unsqueeze(0)
        
        src_nums = []
        
        for feature in range(self.n_num_cols):
            src_nums.append(
                self.num_embedding[feature](src[:, self.n_cat_cols - 1 + feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src_num = torch.cat(src_nums, dim=1)
        #print(src_cat.shape, src_num.shape)
        src = torch.cat((src_cat, src_num), 1)
        src = src.transpose(0, 1)
        
        output = self.transformer_encoder(src).transpose(0, 1)
        output = torch.flatten(output, start_dim=1)
        output = self.decoder(output)
        #output = self.activation(output)
        
        return output

class TransformerModelv4(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, n_features, dropout=0.5):
        super(TransformerModelv4, self).__init__()
        
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.n_features = n_features
        #self.num_embedding = nn.Linear(1, ninp)

        self.embedding = nn.ModuleList()
        
        for feature in range(n_features):
            self.embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        
        self.decoder = nn.Linear(ninp * n_features, 1)
        
        self.activation = nn.Sigmoid()
        
    
    def forward(self, src):
        
        #src = self.num_embedding(src)
        src_nums = []
        
        for feature in range(self.n_features):
            src_nums.append(
                self.embedding[feature](src[:, feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src = torch.cat(src_nums, dim=1)
        src = src.transpose(0, 1)


        output = self.transformer_encoder(src).transpose(0, 1)
        output = torch.flatten(output, start_dim=1)
        output = self.decoder(output)
        #output = self.activation(output)
        
        return output


class MixtureModelv0(nn.Module):

    def __init__(self, ninp, nhead, nhid, nmodels, nfeatures, nclasses, dropout=0.5):
        super(MixtureModelv0, self).__init__()

        self.attention_mechanism = nn.MultiheadAttention(
                                        ninp, 
                                        nhead, 
                                        dropout=dropout
                                    )

                    
        self.nfeatures = nfeatures
        self.nmodels = nmodels
        #self.num_embedding = nn.Linear(1, ninp)

        self.embedding = nn.ModuleList()
        
        for feature in range(nfeatures):
            self.embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        

        self.representation = nn.Sequential(
                                nn.Linear(nfeatures * ninp, nhid),
                                nn.BatchNorm1d(nhid),                          
                                nn.Dropout(dropout)
                            )


        self.model_weighting = nn.Sequential(
                                    nn.Linear(nfeatures, nmodels),
                                    nn.Softmax(dim=-1)
                                )
        
        self.models = nn.ModuleList()
        
        for model in range(nmodels):
            self.models.append(nn.Linear(nhid, nclasses))
        
    def aggregate(self, attn_mat):
        return attn_mat.sum(dim=1)


    def forward(self, src):
        
        #src = self.num_embedding(src)
        src_nums = []
        
        for feature in range(self.nfeatures):
            src_nums.append(
                self.embedding[feature](src[:, feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src = torch.cat(src_nums, dim=1)
        src = src.transpose(0, 1)

        attn_out, attn_mat = self.attention_mechanism(src, src, src)

        attn_out = attn_out.transpose(0, 1).flatten(start_dim=1)
        attn_mat = self.aggregate(attn_mat)

        representation = self.representation(attn_out)

        model_weights = self.model_weighting(attn_mat).unsqueeze(1)

        outputs = []
        
        for model in range(self.nmodels):
            outputs.append(
                self.models[model](representation)
            )

        output = torch.stack(outputs, dim=0).transpose(0, 1)
        output = torch.bmm(model_weights, output).sum(dim=1)

        return output


class MixtureModelv1(nn.Module):

    def __init__(self, ninp, nhead, nhid, nmodels, nfeatures, nclasses, dropout=0.5):
        super(MixtureModelv1, self).__init__()

        self.attention_mechanism = nn.MultiheadAttention(
                                        ninp, 
                                        nhead, 
                                        dropout=dropout
                                    )

                    
        self.nfeatures = nfeatures
        self.nmodels = nmodels
        #self.num_embedding = nn.Linear(1, ninp)

        self.embedding = nn.ModuleList()
        
        for feature in range(nfeatures):
            self.embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        

        self.representation = nn.Sequential(
                                nn.Linear(nfeatures * ninp, nhid),
                                nn.BatchNorm1d(nhid),                          
                                nn.Dropout(dropout)
                            )


        self.model_weighting = nn.Sequential(
                                    nn.Linear(nfeatures, nmodels),
                                    nn.Softmax(dim=-1)
                                )
        
        self.models = nn.ModuleList()

        self.aggregator = nn.Linear(nfeatures, nfeatures)
        
        for model in range(nmodels):
            self.models.append(nn.Linear(nhid, nclasses))
        
    def aggregate(self, attn_mat):
        return self.aggregator(attn_mat).sum(dim=1)


    def forward(self, src):
        
        #src = self.num_embedding(src)
        src_nums = []
        
        for feature in range(self.nfeatures):
            src_nums.append(
                self.embedding[feature](src[:, feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src = torch.cat(src_nums, dim=1)
        src = src.transpose(0, 1)

        attn_out, attn_mat = self.attention_mechanism(src, src, src)

        attn_out = attn_out.transpose(0, 1).flatten(start_dim=1)
        attn_mat = self.aggregate(attn_mat)

        representation = self.representation(attn_out)

        model_weights = self.model_weighting(attn_mat).unsqueeze(1)

        outputs = []
        
        for model in range(self.nmodels):
            outputs.append(
                self.models[model](representation)
            )

        output = torch.stack(outputs, dim=0).transpose(0, 1)
        output = torch.bmm(model_weights, output).sum(dim=1)

        return output


## Helper functions

def build_inference_pipe(cv, model, preprocessing_pipe, dirname=''):
    best_index = np.argmin(cv.cv_results_['rank_test_acc'])
    best_params = cv.cv_results_['params'][best_index]
    best_metric = cv.cv_results_['mean_test_acc'][best_index]

    checkpoint = os.path.join(dirname, '{}_1'.format(best_metric))
    
    print('Loading checkpoint from {}'.format(checkpoint))

    model_params = joblib.load(os.path.join(checkpoint, 'model_params.jl'))

    model.set_params(**model_params)

    model.load_params(
        f_params=os.path.join(checkpoint, 'params.pt'), 
        f_optimizer=os.path.join(checkpoint, 'optimizer.pt'), 
        f_criterion=os.path.join(checkpoint, 'criterion.pt'), 
        f_history=os.path.join(checkpoint, 'history.json')
        )  

    pipe = pipeline.Pipeline([
        ('preprocessing', preprocessing_pipe),
        ('classifier', model)
    ])

    return pipe


def get_num_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()

    return total_params

class ClusterMultitree(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__( self, n_features, n_clusters=2, max_depth=15, random_state=None):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.random_state = random_state
                
        self._clust = None
        self._trees = None
        
        self.build_submodels()
        
    def build_submodels(self):
        self._clust = cluster.KMeans(self.n_clusters, random_state=self.random_state)
        self._trees = [
            tree.DecisionTreeClassifier(
                max_depth=self.max_depth, 
                random_state=self.random_state
            ) for _ in range(self.n_clusters)
        ]
        
    def get_params(self, deep=True):
        return {
            'n_features': self.n_features,
            'n_clusters': self.n_clusters, 
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        self.build_submodels()
        
        return self
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):

        
        attn_matrices = X[:, self.n_features:]
        original_examples = X[:, :self.n_features]
        clust_labels = self._clust.fit_predict(attn_matrices)
        for label in range(self.n_clusters):
            indices = np.where(clust_labels == label)[0]
            self._trees[label].fit(original_examples[indices], y[indices])        
        
        return self
    
    def predict(self, X):

        attn_matrices = X[:, self.n_features:]
        original_examples = X[:, :self.n_features]

        predictions = np.zeros((original_examples.shape[0],))
        
        clust_labels = self._clust.predict(attn_matrices)
        
        for label in range(self.n_clusters):
            indices = np.where(clust_labels == label)[0]
            if indices.shape[0] > 0:
                predictions[indices] = self._trees[label].predict(original_examples[indices])    
        
        return predictions

    def score(self, X, y):
        preds = self.predict(X)
        score_val = metrics.accuracy_score(y, preds)
        return score_val

    