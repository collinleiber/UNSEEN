from clustpy.deep._utils import encode_batchwise, squared_euclidean_distance, predict_batchwise
from clustpy.deep._train_utils import get_default_deep_clustering_initialization
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin
import tqdm
from clustpy.deep.dcn import DCN, _DCN_Module
from clustpy.deep.dkm import DKM, _DKM_Module
from clustpy.deep.dec import DEC, _DEC_Module

"""
GENERIC
"""


def _nn_loss(embedded, centers, centers_differentiable: bool):
    nn_loss = torch.tensor(0.)
    if centers.shape[0] > 2:
        distances = squared_euclidean_distance(embedded, embedded)
        sorted_distances, _ = torch.sort(distances, dim=1)
        kNN = max(1, embedded.shape[0] // centers.shape[0])
        nn_loss = sorted_distances[:, 1:1 + kNN].mean()
        if centers_differentiable:
            fnn_loss = distances.sum() / (distances.shape[0] * (distances.shape[0] - 1))
            nn_loss = nn_loss / fnn_loss
    return nn_loss


def _UNSEEN(X: np.ndarray, n_clusters: int, dying_threshold: float, batch_size: int, pretrain_optimizer_params: dict,
            clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
            optimizer_class: torch.optim.Optimizer, ssl_loss_fn: torch.nn.modules.loss._Loss,
            neural_network: torch.nn.Module | tuple, neural_network_weights: str,
            embedding_size: int, clustering_loss_weight: float, ssl_loss_weight: float,
            custom_dataloaders: tuple, augmentation_invariance: bool, initial_clustering_class: ClusterMixin,
            initial_clustering_params: dict, device: torch.device,
            random_state: np.random.RandomState, base_algorithm: str) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual UNSEEN clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    dying_threshold : float
        threshold for letting a cluster die
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network
    clustering_epochs : int
        number of epochs for the actual clustering procedure
    optimizer_class : torch.optim.Optimizer
        the optimizer class
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    neural_network : torch.nn.Module | tuple
        the input neural network.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network.
    embedding_size : int
        size of the embedding within the neural network
    clustering_loss_weight : float
        weight of the clustering loss
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    device : torch.device
        The device on which to perform the computations
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DCN after the training terminated,
        The cluster centers as identified by DCN after the training terminated,
        The final neural network
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, _, neural_network, _, n_clusters, init_labels, init_centers, _ = get_default_deep_clustering_initialization(
        X, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, ssl_loss_fn,
        neural_network, embedding_size, custom_dataloaders, initial_clustering_class, initial_clustering_params, device,
        random_state, neural_network_weights=neural_network_weights)
    # Setup clustering Module
    if base_algorithm == "DCN":
        clustering_module = _UNSEEN_DCN_Module(init_labels, init_centers, augmentation_invariance).to_device(device)
    elif base_algorithm == "DKM":
        clustering_module = _UNSEEN_DKM_Module(init_centers, [1000], augmentation_invariance).to(device)
    elif base_algorithm == "DEC":
        clustering_module = _UNSEEN_DEC_Module(init_centers, 1, augmentation_invariance).to(device)
    else:
        raise Exception("base_algorithm must be DCN, DKM or DEC")
    # Use DCN optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(neural_network.parameters()) + list(clustering_module.parameters()),
                                **clustering_optimizer_params)
    # DEC Training loop
    clustering_module.fit(neural_network, trainloader, testloader, clustering_epochs, device, optimizer, ssl_loss_fn,
                          clustering_loss_weight, ssl_loss_weight, dying_threshold)
    # Get labels
    clustering_labels = predict_batchwise(testloader, neural_network, clustering_module)
    clustering_centers = clustering_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, neural_network)
    kmeans = KMeans(n_clusters=clustering_centers.shape[0], random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, clustering_labels, clustering_centers, neural_network


"""
DEC
"""


class _UNSEEN_DEC_Module(_DEC_Module):

    def _loss(self, batch: list, neural_network: torch.nn.Module, clustering_loss_weight: float,
              ssl_loss_weight: float, ssl_loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
        """
        Calculate the complete DEC + optional neural network loss.

        Parameters
        ----------
        batch : list
            the minibatch
        neural_network : torch.nn.Module
            the neural network
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the clustering loss
        ssl_loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """
        loss = torch.tensor(0.).to(device)
        # Reconstruction loss is not included in DEC
        if ssl_loss_weight != 0:
            if self.augmentation_invariance:
                ssl_loss, embedded, _, embedded_aug, _ = neural_network.loss_augmentation(batch, ssl_loss_fn, device)
            else:
                ssl_loss, embedded, _ = neural_network.loss(batch, ssl_loss_fn, device)
            loss += ssl_loss_weight * ssl_loss
        else:
            if self.augmentation_invariance:
                aug_data = batch[1].to(device)
                embedded_aug = neural_network.encode(aug_data)
                orig_data = batch[2].to(device)
                embedded = neural_network.encode(orig_data)
            else:
                batch_data = batch[1].to(device)
                embedded = neural_network.encode(batch_data)

        # CLuster loss
        if self.augmentation_invariance:
            cluster_loss = self.dec_augmentation_invariance_loss(embedded, embedded_aug)
        else:
            cluster_loss = self.dec_loss(embedded)
        loss += cluster_loss * clustering_loss_weight

        # Add nearest neighbor loss
        nn_loss = _nn_loss(embedded, self.centers, True)
        loss = loss + clustering_loss_weight * nn_loss

        return loss

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, ssl_loss_fn: torch.nn.modules.loss._Loss,
            clustering_loss_weight: float, ssl_loss_weight: float, dying_threshold: float) -> '_UNSEEN_DEC_Module':
        """
        Trains the _DEC_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        testloader : torch.utils.data.DataLoader
            dataloader to be used for updating the clustering parameters
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        dying_threshold : float
            threshold for letting a cluster die

        Returns
        -------
        self : _UNSEEN_DEC_Module
            this instance of the _UNSEEN_DEC_Module
        """
        init_labels = torch.from_numpy(predict_batchwise(testloader, neural_network, self)).to(device)
        _, init_sizes = torch.unique(init_labels, return_counts=True)
        tbar = tqdm.trange(n_epochs, desc="DEC training")
        for _ in tbar:
            total_loss = 0
            for batch in trainloader:
                loss = self._loss(batch, neural_network, clustering_loss_weight, ssl_loss_weight, ssl_loss_fn,
                                  device)
                total_loss += loss.item()
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                new_labels = torch.from_numpy(predict_batchwise(testloader, neural_network, self)).to(device)
                new_clusters, new_sizes = torch.unique(new_labels, return_counts=True)
                iter_sizes = torch.zeros(init_sizes.shape[0]).to(device)
                for i, clust in enumerate(new_clusters):
                    iter_sizes[clust] = new_sizes[i]

                prop_size = (iter_sizes / init_sizes)
                centers_to_keep = prop_size >= dying_threshold
                if not torch.all(centers_to_keep):
                    center_ids_to_delete = torch.where(~centers_to_keep)[0]
                    samples_in_deleted_cluster = torch.isin(init_labels, center_ids_to_delete)
                    self.centers = torch.nn.Parameter(self.centers[centers_to_keep], requires_grad=True)
                    init_sizes = init_sizes[centers_to_keep]
                    new_labels = predict_batchwise(testloader, neural_network, self)
                    new_labels = torch.from_numpy(new_labels).to(device)

                    labels_of_samples_in_deleted_cluster = new_labels[samples_in_deleted_cluster]
                    unique_labels, unique_sizes = torch.unique(labels_of_samples_in_deleted_cluster,
                                                               return_counts=True)
                    for i in range(len(unique_labels)):
                        init_sizes[unique_labels[i]] += unique_sizes[i]
                    for clust in center_ids_to_delete.tolist()[::-1]:
                        init_labels[init_labels > clust] -= 1
                    init_labels[samples_in_deleted_cluster] = labels_of_samples_in_deleted_cluster
                    assert torch.equal(init_sizes, torch.unique(init_labels, return_counts=True)[
                        1]), "init_sizes: {0}  uniques: {1}".format(init_sizes, torch.unique(init_labels,
                                                                                             return_counts=True)[1])
                    assert torch.sum(init_sizes) == init_labels.shape[0], "size is wrong ({0})".format(
                        torch.sum(init_sizes))

            postfix_str = {"Loss": total_loss, "n_clusters": len(self.centers)}
            tbar.set_postfix(postfix_str)
        return self


class UNSEEN_DEC(DEC):
    """
    The Deep Embedded Clustering (DEC) algorithm.
    First, a neural_network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the network will be optimized using the DEC loss function.

    Parameters
    ----------
    n_clusters_init : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN (default: 35)
    dying_threshold : float
        threshold for letting a cluster die (default: 0.5)
    alpha : float
        alpha value for the prediction (default: 1.0)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    clustering_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dec_labels_ : np.ndarray
        The final DEC labels
    dec_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    neural_network : torch.nn.Module
        The final neural network

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DEC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> dec = DEC(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> dec.fit(data)

    References
    ----------
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis."
    International conference on machine learning. 2016.
    """

    def __init__(self, n_clusters_init: int = 35, dying_threshold: float = 0.5, alpha: float = 1.0,
                 batch_size: int = 256,
                 pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, clustering_loss_weight: float = 1., custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = None, device: torch.device = None,
                 random_state: np.random.RandomState | int = None):
        super().__init__(n_clusters_init, alpha, batch_size, pretrain_optimizer_params, clustering_optimizer_params,
                         pretrain_epochs, clustering_epochs, optimizer_class, ssl_loss_fn, neural_network,
                         neural_network_weights, embedding_size, clustering_loss_weight, custom_dataloaders,
                         augmentation_invariance, initial_clustering_class, initial_clustering_params, device,
                         random_state)
        self.dying_threshold = dying_threshold

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'UNSEEN_DEC':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : UNSEEN_DEC
            this instance of the UNSEEN_DEC algorithm
        """
        kmeans_labels, kmeans_centers, dec_labels, dec_centers, neural_network = _UNSEEN(X, self.n_clusters,
                                                                                         self.dying_threshold,
                                                                                         self.batch_size,
                                                                                         self.pretrain_optimizer_params,
                                                                                         self.clustering_optimizer_params,
                                                                                         self.pretrain_epochs,
                                                                                         self.clustering_epochs,
                                                                                         self.optimizer_class,
                                                                                         self.ssl_loss_fn,
                                                                                         self.neural_network,
                                                                                         self.neural_network_weights,
                                                                                         self.embedding_size,
                                                                                         self.clustering_loss_weight,
                                                                                         self.ssl_loss_weight,
                                                                                         self.custom_dataloaders,
                                                                                         self.augmentation_invariance,
                                                                                         self.initial_clustering_class,
                                                                                         self.initial_clustering_params,
                                                                                         self.device, self.random_state,
                                                                                         "DEC")
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dec_labels_ = dec_labels
        self.dec_cluster_centers_ = dec_centers
        self.neural_network = neural_network
        self.n_clusters = kmeans_centers.shape[0]
        return self


"""
DKM
"""


class _UNSEEN_DKM_Module(_DKM_Module):

    def _loss(self, batch: list, alpha: float, neural_network: torch.nn.Module, clustering_loss_weight: float,
              ssl_loss_weight: float, ssl_loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
        """
        Calculate the complete DKM + neural network loss.

        Parameters
        ----------
        batch : list
            the minibatch
        alpha : float
            the alpha value
        neural_network : torch.nn.Module
            the neural network
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DKM loss
        """
        # Calculate combined total loss
        if self.augmentation_invariance:
            # Calculate ssl loss
            ssl_loss, embedded, _, embedded_aug, _ = neural_network.loss_augmentation(batch, ssl_loss_fn, device)
            # Calculate clustering loss
            cluster_loss = self.dkm_augmentation_invariance_loss(embedded, embedded_aug, alpha)
        else:
            # Calculate ssl loss
            ssl_loss, embedded, _ = neural_network.loss(batch, ssl_loss_fn, device)
            # Calculate clustering loss
            cluster_loss = self.dkm_loss(embedded, alpha)
        loss = ssl_loss_weight * ssl_loss + cluster_loss * clustering_loss_weight

        # Add nearest neighbor loss
        nn_loss = _nn_loss(embedded, self.centers, True)
        loss = loss + clustering_loss_weight * nn_loss

        return loss

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
            n_epochs: int, device: torch.device, optimizer: torch.optim.Optimizer,
            ssl_loss_fn: torch.nn.modules.loss._Loss,
            clustering_loss_weight: float, ssl_loss_weight: float, dying_threshold: float) -> '_UNSEEN_DKM_Module':
        """
        Trains the _DKM_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        testloader : torch.utils.data.DataLoader
            dataloader to be used for updating the clustering parameters
        n_epochs : int
            number of epochs for the clustering procedure.
            The total number of epochs therefore corresponds to: len(alphas)*n_epochs
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        dying_threshold : float
            threshold for letting a cluster die

        Returns
        -------
        self : _UNSEEN_DKM_Module
            this instance of the _UNSEEN_DKM_Module
        """
        init_labels = torch.from_numpy(predict_batchwise(testloader, neural_network, self)).to(device)
        _, init_sizes = torch.unique(init_labels, return_counts=True)
        tbar = tqdm.tqdm(total=n_epochs * len(self.alphas), desc="DKM training")
        for alpha in self.alphas:
            for _ in range(n_epochs):
                total_loss = 0
                for batch in trainloader:
                    loss = self._loss(batch, alpha, neural_network, clustering_loss_weight, ssl_loss_weight,
                                      ssl_loss_fn, device)
                    total_loss += loss.item()
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    new_labels = torch.from_numpy(predict_batchwise(testloader, neural_network, self)).to(device)
                    new_clusters, new_sizes = torch.unique(new_labels, return_counts=True)
                    iter_sizes = torch.zeros(init_sizes.shape[0]).to(device)
                    for i, clust in enumerate(new_clusters):
                        iter_sizes[clust] = new_sizes[i]

                    prop_size = (iter_sizes / init_sizes)
                    centers_to_keep = prop_size >= dying_threshold
                    if not torch.all(centers_to_keep):
                        center_ids_to_delete = torch.where(~centers_to_keep)[0]
                        samples_in_deleted_cluster = torch.isin(init_labels, center_ids_to_delete)
                        self.centers = torch.nn.Parameter(self.centers[centers_to_keep], requires_grad=True)
                        init_sizes = init_sizes[centers_to_keep]
                        new_labels = predict_batchwise(testloader, neural_network, self)
                        new_labels = torch.from_numpy(new_labels).to(device)

                        labels_of_samples_in_deleted_cluster = new_labels[samples_in_deleted_cluster]
                        unique_labels, unique_sizes = torch.unique(labels_of_samples_in_deleted_cluster,
                                                                   return_counts=True)
                        for i in range(len(unique_labels)):
                            init_sizes[unique_labels[i]] += unique_sizes[i]
                        for clust in center_ids_to_delete.tolist()[::-1]:
                            init_labels[init_labels > clust] -= 1
                        init_labels[samples_in_deleted_cluster] = labels_of_samples_in_deleted_cluster
                        assert torch.equal(init_sizes, torch.unique(init_labels, return_counts=True)[
                            1]), "init_sizes: {0}  uniques: {1}".format(init_sizes, torch.unique(init_labels,
                                                                                                 return_counts=True)[1])
                        assert torch.sum(init_sizes) == init_labels.shape[0], "size is wrong ({0})".format(
                            torch.sum(init_sizes))
                postfix_str = {"Loss": total_loss, "n_clusters": len(self.centers)}
                tbar.set_postfix(postfix_str)
                tbar.update()
        return self


class UNSEEN_DKM(DKM):
    """
    The Deep k-Means (DKM) algorithm.
    First, a neural network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the network will be optimized using the DKM loss function.

    Parameters
    ----------
    n_clusters_init : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN (default: 35)
    dying_threshold : float
        threshold for letting a cluster die (default: 0.5)
    alphas : tuple
        tuple of different alpha values used for the prediction.
        Small values close to 0 are equivalent to homogeneous assignments to all clusters. Large values simulate a clear assignment as with kMeans.
        If None, the default calculation of the paper will be used.
        This is equal to \alpha_{i+1}=2^{1/log(i)^2}*\alpha_i with \alpha_1=0.1 and maximum i=40.
        Alpha can also be a tuple with (None, \alpha_1, maximum i) (default: (1000))
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 50)
    clustering_epochs : int
        number of epochs for each alpha value for the actual clustering procedure.
        The total number of clustering epochs therefore corresponds to: len(alphas)*clustering_epochs (default: 100)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    clustering_loss_weight : float
        weight of the clustering loss (default: 1.0)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dkm_labels_ : np.ndarray
        The final DKM labels
    dkm_cluster_centers_ : np.ndarray
        The final DKM cluster centers
    neural_network : torch.nn.Module
        The final neural network

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DKM
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> dkm = DKM(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> dkm.fit(data)

    References
    ----------
    Fard, Maziar Moradi, Thibaut Thonet, and Eric Gaussier. "Deep k-means: Jointly clustering with k-means and learning representations."
    Pattern Recognition Letters 138 (2020): 185-192.
    """

    def __init__(self, n_clusters_init: int = 35, dying_threshold: float = 0.5, alphas: tuple = (1000),
                 batch_size: int = 256,
                 pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 50, clustering_epochs: int = 100,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, clustering_loss_weight: float = 1., ssl_loss_weight: float = 1.,
                 custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 initial_clustering_class: ClusterMixin = KMeans, initial_clustering_params: dict = None,
                 device: torch.device = None, random_state: np.random.RandomState | int = None):
        super().__init__(n_clusters_init, alphas, batch_size, pretrain_optimizer_params, clustering_optimizer_params,
                         pretrain_epochs, clustering_epochs, optimizer_class, ssl_loss_fn, neural_network,
                         neural_network_weights, embedding_size, clustering_loss_weight, ssl_loss_weight,
                         custom_dataloaders, augmentation_invariance, initial_clustering_class,
                         initial_clustering_params, device, random_state)
        self.dying_threshold = dying_threshold

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DKM':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : DKM
            this instance of the DKM algorithm
        """
        kmeans_labels, kmeans_centers, dkm_labels, dkm_centers, neural_network = _UNSEEN(X, self.n_clusters,
                                                                                         self.dying_threshold,
                                                                                         self.batch_size,
                                                                                         self.pretrain_optimizer_params,
                                                                                         self.clustering_optimizer_params,
                                                                                         self.pretrain_epochs,
                                                                                         self.clustering_epochs,
                                                                                         self.optimizer_class,
                                                                                         self.ssl_loss_fn,
                                                                                         self.neural_network,
                                                                                         self.neural_network_weights,
                                                                                         self.embedding_size,
                                                                                         self.clustering_loss_weight,
                                                                                         self.ssl_loss_weight,
                                                                                         self.custom_dataloaders,
                                                                                         self.augmentation_invariance,
                                                                                         self.initial_clustering_class,
                                                                                         self.initial_clustering_params,
                                                                                         self.device,
                                                                                         self.random_state, "DKM")
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dkm_labels_ = dkm_labels
        self.dkm_cluster_centers_ = dkm_centers
        self.neural_network = neural_network
        self.n_clusters = kmeans_centers.shape[0]
        return self


"""
DCN
"""


class _UNSEEN_DCN_Module(_DCN_Module):

    def _loss(self, batch: list, neural_network: torch.nn.Module, ssl_loss_fn: torch.nn.modules.loss._Loss,
              ssl_loss_weight: float, clustering_loss_weight: float, device: torch.device) -> torch.Tensor:
        """
        Calculate the complete DCN + neural network loss.

        Parameters
        ----------
        batch : list
            the minibatch
        neural_network : torch.nn.Module
            the neural network
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DCN loss
        """
        # compute self-supervised loss
        if self.augmentation_invariance:
            ssl_loss, embedded, _, embedded_aug, _ = neural_network.loss_augmentation(batch, ssl_loss_fn, device)
        else:
            ssl_loss, embedded, _ = neural_network.loss(batch, ssl_loss_fn, device)

        # compute cluster loss
        labels = self.labels[batch[0]]
        cluster_loss = self.dcn_loss(embedded, labels)
        if self.augmentation_invariance:
            # assign augmented samples to the same cluster as original samples
            cluster_loss_aug = self.dcn_loss(embedded_aug, labels)
            cluster_loss = (cluster_loss + cluster_loss_aug) / 2

        # compute total loss
        loss = ssl_loss_weight * ssl_loss + 0.5 * clustering_loss_weight * cluster_loss

        # Add nearest neighbor loss
        nn_loss = _nn_loss(embedded, self.centers, False)
        loss = loss + 0.5 * clustering_loss_weight * nn_loss

        return loss

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader, n_epochs: int, device: torch.device,
            optimizer: torch.optim.Optimizer, ssl_loss_fn: torch.nn.modules.loss._Loss, clustering_loss_weight: float,
            ssl_loss_weight: float, dying_threshold: float) -> '_UNSEEN_DCN_Module':
        """
        Trains the _DCN_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        testloader : torch.utils.data.DataLoader
            dataloader to be used for updating the clustering parameters
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        dying_threshold : float
            threshold for letting a cluster die

        Returns
        -------
        self : _UNSEEN_DCN_Module
            this instance of the _UNSEEN_DCN_Module
        """
        # DCN training loop
        init_labels = self.labels.clone()
        _, init_sizes = torch.unique(self.labels, return_counts=True)
        tbar = tqdm.trange(n_epochs, desc="DCN training")
        for _ in tbar:
            # Update Network
            total_loss = 0
            for batch in trainloader:
                loss = self._loss(batch, neural_network, ssl_loss_fn, ssl_loss_weight, clustering_loss_weight,
                                  device)
                total_loss += loss.item()
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Update Assignments and Centroids
                with torch.no_grad():
                    if self.augmentation_invariance:
                        # Convention is that the augmented sample is at the first position and the original one at the second position
                        # We only use the original sample for updating the centroids and assignments
                        batch_data = batch[2].to(device)
                    else:
                        batch_data = batch[1].to(device)
                    embedded = neural_network.encode(batch_data)
                    labels_new = self.predict_hard(embedded)
                    self.labels[batch[0]] = labels_new

                    ## update centroids [on gpu] About 40 seconds for 1000 iterations
                    ## No overhead from loading between gpu and cpu
                    # counts = cluster_module.update_centroid(embedded, counts, s)

                    # update centroids [on cpu] About 30 Seconds for 1000 iterations
                    # with additional overhead from loading between gpu and cpu
                    centers, counts = self.update_centroids(embedded.cpu(), labels_new.cpu())
                    self.centers = centers.to(device)
                    self.counts = counts

            with torch.no_grad():
                new_labels = self.labels
                new_clusters, new_sizes = torch.unique(new_labels, return_counts=True)
                iter_sizes = torch.zeros(init_sizes.shape[0]).to(device)
                for i, clust in enumerate(new_clusters):
                    iter_sizes[clust] = new_sizes[i]

                prop_size = (iter_sizes / init_sizes)
                centers_to_keep = prop_size >= dying_threshold
                if not torch.all(centers_to_keep):
                    center_ids_to_delete = torch.where(~centers_to_keep)[0]
                    samples_in_deleted_cluster = torch.isin(init_labels, center_ids_to_delete)
                    self.centers = self.centers[centers_to_keep].to(device)
                    init_sizes = init_sizes[centers_to_keep]
                    new_labels = predict_batchwise(testloader, neural_network, self)
                    new_labels = torch.from_numpy(new_labels).to(device)
                    self.labels = new_labels

                    labels_of_samples_in_deleted_cluster = new_labels[samples_in_deleted_cluster]
                    unique_labels, unique_sizes = torch.unique(labels_of_samples_in_deleted_cluster,
                                                               return_counts=True)
                    for i in range(len(unique_labels)):
                        init_sizes[unique_labels[i]] += unique_sizes[i]
                    for clust in center_ids_to_delete.tolist()[::-1]:
                        init_labels[init_labels > clust] -= 1
                    init_labels[samples_in_deleted_cluster] = labels_of_samples_in_deleted_cluster
                    assert torch.equal(init_sizes, torch.unique(init_labels, return_counts=True)[
                        1]), "init_sizes: {0}  uniques: {1}".format(init_sizes, torch.unique(init_labels,
                                                                                             return_counts=True)[1])
                    assert torch.sum(init_sizes) == init_labels.shape[0], "size is wrong ({0})".format(
                        torch.sum(init_sizes))

            postfix_str = {"Loss": total_loss, "n_clusters": len(self.centers)}
            tbar.set_postfix(postfix_str)
        return self


class UNSEEN_DCN(DCN):
    """
    The Deep Clustering Network (DCN) algorithm.
    First, a neural network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DCN loss function.

    Parameters
    ----------
    n_clusters_init : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN (default: 35)
    dying_threshold : float
        threshold for letting a cluster die (default: 0.5)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    clustering_loss_weight : float
        weight of the clustering loss (default: 0.05)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dcn_labels_ : np.ndarray
        The final DCN labels
    dcn_cluster_centers_ : np.ndarray
        The final DCN cluster centers
    neural_network : torch.nn.Module
        The final neural network

    References
    ----------
    Yang, Bo, et al. "Towards k-means-friendly spaces: Simultaneous deep learning and clustering."
    international conference on machine learning. PMLR, 2017.
    """

    def __init__(self, n_clusters_init: int = 35, dying_threshold: float = 0.5, batch_size: int = 256,
                 pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 50,
                 clustering_epochs: int = 50, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), clustering_loss_weight: float = 0.05,
                 ssl_loss_weight: float = 1.0, neural_network: torch.nn.Module | tuple = None,
                 neural_network_weights: str = None, embedding_size: int = 10, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = None, device: torch.device = None,
                 random_state: np.random.RandomState | int = None):
        super().__init__(n_clusters_init, batch_size, pretrain_optimizer_params, clustering_optimizer_params,
                         pretrain_epochs, clustering_epochs, optimizer_class, ssl_loss_fn, clustering_loss_weight,
                         ssl_loss_weight, neural_network, neural_network_weights, embedding_size, custom_dataloaders,
                         augmentation_invariance, initial_clustering_class, initial_clustering_params, device,
                         random_state)
        self.dying_threshold = dying_threshold

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'UNSEEN_DCN':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : UNSEEN_DCN
            this instance of the UNSEEN_DCN algorithm
        """
        kmeans_labels, kmeans_centers, dcn_labels, dcn_centers, neural_network = _UNSEEN(X, self.n_clusters,
                                                                                         self.dying_threshold,
                                                                                         self.batch_size,
                                                                                         self.pretrain_optimizer_params,
                                                                                         self.clustering_optimizer_params,
                                                                                         self.pretrain_epochs,
                                                                                         self.clustering_epochs,
                                                                                         self.optimizer_class,
                                                                                         self.ssl_loss_fn,
                                                                                         self.neural_network,
                                                                                         self.neural_network_weights,
                                                                                         self.embedding_size,
                                                                                         self.clustering_loss_weight,
                                                                                         self.ssl_loss_weight,
                                                                                         self.custom_dataloaders,
                                                                                         self.augmentation_invariance,
                                                                                         self.initial_clustering_class,
                                                                                         self.initial_clustering_params,
                                                                                         self.device,
                                                                                         self.random_state, "DCN")
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dcn_labels_ = dcn_labels
        self.dcn_cluster_centers_ = dcn_centers
        self.neural_network = neural_network
        self.n_clusters = kmeans_centers.shape[0]
        return self


"""
EXPERIMENTS
"""

from clustpy.utils import evaluate_multiple_datasets, EvaluationDataset, EvaluationMetric, EvaluationAlgorithm
from clustpy.metrics import unsupervised_clustering_accuracy as acc, fair_normalized_mutual_information as fnmi
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from sklearn.datasets import make_blobs
from clustpy.deep import get_trained_network, get_dataloader
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from clustpy.deep._utils import embedded_kmeans_prediction
from clustpy.data import load_mnist, load_optdigits, load_kmnist, load_usps, load_pendigits, load_fmnist
from sklearn.utils import check_random_state

EXPERIMENT_PATH = "PATH/UNSEEN/"


class AEKmeans():

    def __init__(self, n_clusters, neural_network=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None, neural_network_weights: str = None):
        self.n_clusters = n_clusters
        self.neural_network = neural_network
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders
        self.neural_network_weights = neural_network_weights

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
        _, _, _, _, _, _, _, init_labels, init_centers, _ = get_default_deep_clustering_initialization(
            X, self.n_clusters, self.batch_size, {"lr": 1e-3}, 100, torch.optim.Adam, torch.nn.MSELoss(),
            self.neural_network, 10, self.custom_dataloaders, KMeans, {}, torch.device('cpu'), self.random_state,
            neural_network_weights=self.neural_network_weights)
        self.labels_ = init_labels
        self.cluster_centers_ = init_centers
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dataloader = get_dataloader(X, self.batch_size, False, False)
        ae = self.neural_network.to(torch.device('cpu'))
        predicted_labels = embedded_kmeans_prediction(dataloader, self.cluster_centers_, ae)
        return predicted_labels


def image_normalize(X):
    X = (X - np.mean(X)) / np.std(X)
    return X


def tabular_normalize(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    return X


def blobs_experiment():
    metrics = [EvaluationMetric("ACC", acc), EvaluationMetric("ARI", ari), EvaluationMetric("NMI", nmi),
               EvaluationMetric("FNMI", fnmi)]
    datasets = [EvaluationDataset("k_{0}".format(n_clusters), make_blobs,
                                  data_loader_params={"n_samples": 10000, "n_features": 100, "centers": n_clusters,
                                                      "random_state": 1}, preprocess_methods=tabular_normalize) for
                n_clusters
                in [5, 7, 10, 15, 20, 25, 30]]
    algorithms = [EvaluationAlgorithm("UNSEEN+DCN", UNSEEN_DCN,
                                      params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                              "clustering_epochs": 150}),
                  EvaluationAlgorithm("UNSEEN+DEC", UNSEEN_DEC,
                                      params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                              "clustering_epochs": 150}),
                  EvaluationAlgorithm("UNSEEN+DKM", UNSEEN_DKM,
                                      params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                              "clustering_epochs": 150})
                  ]
    evaluate_multiple_datasets(datasets, algorithms, metrics, add_n_clusters=True,
                               save_path=EXPERIMENT_PATH + "blobs_experiments.csv",
                               save_intermediate_results=True,
                               random_state=1)


def quantitative_experiments():
    neural_network = (FeedforwardAutoencoder, None)
    iteration_specific_params = {
        ("Optdigits", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/Optdigits_{0}.pth".format(i) for i in
                                                  range(10)],
        ("MNIST", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/MNIST_{0}.pth".format(i) for i in
                                              range(10)],
        ("USPS", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/USPS_{0}.pth".format(i) for i in
                                             range(10)],
        ("FMNIST", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/FMNIST_{0}.pth".format(i) for i in
                                               range(10)],
        ("KMNIST", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/KMNIST_{0}.pth".format(i) for i in
                                               range(10)],
        ("Pendigits", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/Pendigits_{0}.pth".format(i) for i in
                                                  range(10)]
    }
    metrics = [EvaluationMetric("ACC", acc), EvaluationMetric("ARI", ari), EvaluationMetric("NMI", nmi),
               EvaluationMetric("FNMI", fnmi)]
    algorithms = [
        EvaluationAlgorithm("AE+Kmeans", AEKmeans,
                            params={"n_clusters": None, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("DCN", DCN,
                            params={"n_clusters": None, "pretrain_epochs": 100, "clustering_epochs": 150,
                                    "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("DEC", DEC,
                            params={"n_clusters": None, "pretrain_epochs": 100, "clustering_epochs": 150,
                                    "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("DKM", DKM,
                            params={"n_clusters": None, "pretrain_epochs": 100, "clustering_epochs": 150,
                                    "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params)
    ]
    datasets = [
        EvaluationDataset("Optdigits", load_optdigits, preprocess_methods=image_normalize),
        EvaluationDataset("MNIST", load_mnist, preprocess_methods=image_normalize),
        EvaluationDataset("USPS", load_usps, preprocess_methods=image_normalize),
        EvaluationDataset("FMNIST", load_fmnist, preprocess_methods=image_normalize),
        EvaluationDataset("KMNIST", load_kmnist, preprocess_methods=image_normalize),
        EvaluationDataset("Pendigits", load_pendigits, preprocess_methods=tabular_normalize)
    ]
    evaluate_multiple_datasets(datasets, algorithms, metrics, add_n_clusters=True,
                               save_path=EXPERIMENT_PATH + "quantitative_experiments.csv",
                               save_intermediate_results=True,
                               random_state=1)


def dying_threshold():
    neural_network = (FeedforwardAutoencoder, None)
    iteration_specific_params = {
        ("Optdigits", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/Optdigits_{0}.pth".format(i) for i in
                                                  range(10)],
        ("MNIST", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/MNIST_{0}.pth".format(i) for i in
                                              range(10)],
        ("USPS", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/USPS_{0}.pth".format(i) for i in
                                             range(10)],
        ("FMNIST", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/FMNIST_{0}.pth".format(i) for i in
                                               range(10)],
        ("KMNIST", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/KMNIST_{0}.pth".format(i) for i in
                                               range(10)],
        ("Pendigits", "neural_network_weights"): [EXPERIMENT_PATH + "AEs/Pendigits_{0}.pth".format(i) for i in
                                                  range(10)]
    }
    metrics = [EvaluationMetric("ACC", acc), EvaluationMetric("ARI", ari), EvaluationMetric("NMI", nmi),
               EvaluationMetric("FNMI", fnmi)]
    algorithms = [
        EvaluationAlgorithm("UNSEEN+DCN_01", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.1, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN_02", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.2, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN_03", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.3, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN_04", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.4, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN_06", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.6, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN_07", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.7, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN_08", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.8, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DCN_09", UNSEEN_DCN,
                            params={"n_clusters_init": 35, "dying_threshold": 0.9, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_01", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.1, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_02", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.2, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_03", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.3, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_04", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.4, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_06", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.6, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_07", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.7, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_08", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.8, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DEC_09", UNSEEN_DEC,
                            params={"n_clusters_init": 35, "dying_threshold": 0.9, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_01", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.1, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_02", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.2, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_03", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.3, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_04", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.4, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_06", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.6, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_07", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.7, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_08", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.8, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
        EvaluationAlgorithm("UNSEEN+DKM_09", UNSEEN_DKM,
                            params={"n_clusters_init": 35, "dying_threshold": 0.9, "pretrain_epochs": 100,
                                    "clustering_epochs": 150, "neural_network": neural_network},
                            iteration_specific_params=iteration_specific_params),
    ]
    datasets = [
        EvaluationDataset("Optdigits", load_optdigits, preprocess_methods=image_normalize),
        EvaluationDataset("MNIST", load_mnist, preprocess_methods=image_normalize),
        EvaluationDataset("USPS", load_usps, preprocess_methods=image_normalize),
        EvaluationDataset("FMNIST", load_fmnist, preprocess_methods=image_normalize),
        EvaluationDataset("KMNIST", load_kmnist, preprocess_methods=image_normalize),
        EvaluationDataset("Pendigits", load_pendigits, preprocess_methods=tabular_normalize),
    ]
    evaluate_multiple_datasets(datasets, algorithms, metrics, add_n_clusters=True,
                               save_path=EXPERIMENT_PATH + "dying_threshold.csv",
                               save_intermediate_results=True,
                               random_state=1)


def not_all_mnist_clusters_experiment():
    def _selected_dataloader(data_loader, n_clusters):
        X, L = data_loader(return_X_y=True)
        X = image_normalize(X)
        X = X[L <= n_clusters]
        L = L[L <= n_clusters]
        return X, L

    metrics = [EvaluationMetric("ACC", acc), EvaluationMetric("ARI", ari), EvaluationMetric("NMI", nmi),
               EvaluationMetric("FNMI", fnmi)]
    datasets = [EvaluationDataset("MNIST_{0}".format(n_clusters), _selected_dataloader,
                                  data_loader_params={"data_loader": load_mnist, "n_clusters": n_clusters}) for
                n_clusters in range(0, 9)]
    algorithms = [EvaluationAlgorithm("UNSEEN+DCN", UNSEEN_DCN,
                                      params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                              "clustering_epochs": 150}),
                  EvaluationAlgorithm("UNSEEN+DEC", UNSEEN_DEC,
                                      params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                              "clustering_epochs": 150}),
                  EvaluationAlgorithm("UNSEEN+DKM", UNSEEN_DKM,
                                      params={"n_clusters_init": 35, "dying_threshold": 0.5, "pretrain_epochs": 100,
                                              "clustering_epochs": 150})
                  ]
    evaluate_multiple_datasets(datasets, algorithms, metrics, add_n_clusters=True,
                               save_path=EXPERIMENT_PATH + "mnist_different_k_experiments.csv",
                               save_intermediate_results=True,
                               random_state=1)


def create_pretrained_aes():
    datasets = [
        ("Optdigits", load_optdigits, True), ("MNIST", load_mnist, True), ("USPS", load_usps, True),
        ("FMNIST", load_fmnist, True), ("KMNIST", load_kmnist, True), ("Pendigits", load_pendigits, False)
    ]
    for data_name, data_laoder, is_image in datasets:
        X, L = data_laoder(return_X_y=True)
        if is_image:
            X = image_normalize(X)
        else:
            X = tabular_normalize(X)
        for i in range(10):
            network = get_trained_network(data=X, n_epochs=100, batch_size=256, random_state=i)
            network.save_parameters(EXPERIMENT_PATH + "AEs/{0}_{1}.pth".format(data_name, i))


if __name__ == "__main__":
    create_pretrained_aes()
    quantitative_experiments()
    # blobs_experiment()
    # not_all_mnist_clusters_experiment()
    # dying_threshold()
