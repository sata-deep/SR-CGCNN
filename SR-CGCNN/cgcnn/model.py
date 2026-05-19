from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    # ****** Modified by Satadeep Bhattacharjee ****
    def __init__(self, atom_fea_len, nbr_fea_len, batch_norm=True):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        # ****** Modified by Satadeep Bhattacharjee ****
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
            self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    # ****** Modified by Satadeep Bhattacharjee ****
    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, bn1=None, bn2=None):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        bn1: nn.BatchNorm1d, optional
          Batch normalization module for the gated neighbor features.
        bn2: nn.BatchNorm1d, optional
          Batch normalization module for the summed neighbor features.

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        # ****** Modified by Satadeep Bhattacharjee ****
        if bn1 is None:
            bn1 = self.bn1
        if bn2 is None:
            bn2 = self.bn2
        total_gated_fea = bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 # ****** Modified by Satadeep Bhattacharjee ****
                 classification=False, conv_mode="standard",
                 n_recurrent_steps=None, recurrent_layer_norm=False,
                 recurrent_learnable_alpha=False,
                 return_loop_outputs=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        # ****** Modified by Satadeep Bhattacharjee ****
        if conv_mode not in ("standard", "shared_recurrent"):
            raise ValueError("conv_mode must be 'standard' or "
                             "'shared_recurrent'")
        if n_conv < 1:
            raise ValueError("n_conv must be at least 1")
        if conv_mode == "shared_recurrent" and n_recurrent_steps is None:
            n_recurrent_steps = n_conv
        if n_recurrent_steps is not None and n_recurrent_steps < 1:
            raise ValueError("n_recurrent_steps must be at least 1")

        self.classification = classification
        # ****** Modified by Satadeep Bhattacharjee ****
        self.conv_mode = conv_mode
        self.n_conv = n_conv
        self.n_recurrent_steps = n_recurrent_steps
        self.recurrent_layer_norm = recurrent_layer_norm
        self.recurrent_learnable_alpha = recurrent_learnable_alpha
        self.return_loop_outputs = return_loop_outputs
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # ****** Modified by Satadeep Bhattacharjee ****
        if self.conv_mode == "standard":
            self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                        nbr_fea_len=nbr_fea_len)
                                        for _ in range(n_conv)])
        else:
            self.shared_conv = ConvLayer(atom_fea_len=atom_fea_len,
                                         nbr_fea_len=nbr_fea_len,
                                         batch_norm=False)
            self.recurrent_bn1s = nn.ModuleList([
                nn.BatchNorm1d(2*atom_fea_len)
                for _ in range(self.n_recurrent_steps)
            ])
            self.recurrent_bn2s = nn.ModuleList([
                nn.BatchNorm1d(atom_fea_len)
                for _ in range(self.n_recurrent_steps)
            ])
            if self.recurrent_layer_norm:
                self.recurrent_norm = nn.LayerNorm(atom_fea_len)
            if self.recurrent_learnable_alpha:
                self.recurrent_alpha = nn.Parameter(
                    torch.ones(self.n_recurrent_steps))
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        # ****** Modified by Satadeep Bhattacharjee ****
        loop_outputs = []
        if self.conv_mode == "standard":
            for conv_func in self.convs:
                atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
                if self.return_loop_outputs:
                    loop_outputs.append(atom_fea)
        elif self.conv_mode == "shared_recurrent":
            for t in range(self.n_recurrent_steps):
                new_atom_fea = self.shared_conv(atom_fea, nbr_fea,
                                                nbr_fea_idx,
                                                bn1=self.recurrent_bn1s[t],
                                                bn2=self.recurrent_bn2s[t])
                if self.recurrent_learnable_alpha:
                    alpha_t = torch.sigmoid(self.recurrent_alpha[t])
                    atom_fea = (1.0 - alpha_t) * atom_fea + \
                        alpha_t * new_atom_fea
                else:
                    atom_fea = new_atom_fea

                if self.recurrent_layer_norm:
                    atom_fea = self.recurrent_norm(atom_fea)
                if self.return_loop_outputs:
                    loop_outputs.append(atom_fea)
        else:
            raise ValueError("Unknown conv_mode: {}".format(self.conv_mode))
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        # ****** Modified by Satadeep Bhattacharjee ****
        if self.return_loop_outputs:
            return out, loop_outputs
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

    # ****** Modified by Satadeep Bhattacharjee ****
    def count_conv_parameters(self):
        """Count trainable parameters in convolution modules only."""
        if self.conv_mode == "standard":
            modules = self.convs
        else:
            modules = [self.shared_conv, self.recurrent_bn1s,
                       self.recurrent_bn2s]
        return sum(p.numel() for module in modules
                   for p in module.parameters() if p.requires_grad)

    def count_total_parameters(self):
        """Count all trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary_dict(self):
        return {
            "conv_mode": self.conv_mode,
            "n_conv": self.n_conv,
            "n_recurrent_steps": self.n_recurrent_steps,
            "recurrent_layer_norm": self.recurrent_layer_norm,
            "recurrent_learnable_alpha": self.recurrent_learnable_alpha,
            "recurrent_batch_norm": "per_step"
            if self.conv_mode == "shared_recurrent" else "per_layer",
            "total_parameters": self.count_total_parameters(),
            "conv_parameters": self.count_conv_parameters()
        }
