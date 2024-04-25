from .registry import register_model
from .gemnet_oc_ocp import GemNetOC

@register_model
def gemnet_oc_small_qm9(
        ## not used, to adapt the model initialization in main.py
        irreps_in=None,
        radius=None,
        num_basis=None,
        out_channels=None,  
        task_mean=None,
        task_std=None,
        atomref=None,
        **kwargs):      
    model = GemNetOC(
        num_atoms=None,
        bond_feat_dim=None,
        num_targets=1, 
        num_spherical=7,
        num_radial=128,
        num_blocks=4,
        emb_size_atom=256,
        emb_size_edge=512,
        emb_size_trip_in=64,
        emb_size_trip_out=64,
        emb_size_quad_in=32,
        emb_size_quad_out=32,
        emb_size_aint_in=64,
        emb_size_aint_out=64,
        emb_size_rbf=16,
        emb_size_cbf=16,
        emb_size_sbf=32,
        num_before_skip=2,
        num_after_skip=2,
        num_concat=1,
        num_atom=3,
        num_output_afteratom=3,
        cutoff=12,
        cutoff_qint=12,
        cutoff_aeaint=12,
        cutoff_aint=12,
        max_neighbors=30,
        max_neighbors_qint=8,
        max_neighbors_aeaint=20,
        max_neighbors_aint=1000,
        rbf={"name": "gaussian"},
        envelope={"name":"polynomial", "exponent":5},
        cbf={"name": "spherical_harmonics"},
        sbf={"name": "legendre_outer"},
        extensive=True,
        output_init="HeOrthogonal",
        activation="silu",
        scale_file=None,
        quad_interaction=True,
        atom_edge_interaction=True,
        edge_atom_interaction=True,
        atom_interaction=True,
        regress_forces=False,
        num_atom_emb_layers=2,
        num_global_out_layers=2,
        qint_tags=[1,2]
    )   
    return model


@register_model
def gemnet_oc_large_qm9(
        ## not used, to adapt the model initialization in main.py
        irreps_in=None,
        radius=None,
        num_basis=None,
        out_channels=None,  
        task_mean=None,
        task_std=None,
        atomref=None,
        **kwargs):
    model = GemNetOC(
        num_atoms=None,
        bond_feat_dim=None,
        num_targets=1, 
        num_spherical=7,
        num_radial=128,
        num_blocks=6,           ## indicate larger param 
        emb_size_atom=256,
        emb_size_edge=1024,     ##
        emb_size_trip_in=64,   
        emb_size_trip_out=128,  ##
        emb_size_quad_in=64,    ##
        emb_size_quad_out=32,
        emb_size_aint_in=64,
        emb_size_aint_out=64,
        emb_size_rbf=32,        ##
        emb_size_cbf=16,
        emb_size_sbf=64,        ##
        num_before_skip=2,
        num_after_skip=2,
        num_concat=4,           ##
        num_atom=3,             
        num_output_afteratom=3, 
        cutoff=12,
        cutoff_qint=12,
        cutoff_aeaint=12,
        cutoff_aint=12,
        max_neighbors=30,
        max_neighbors_qint=8,
        max_neighbors_aeaint=20,
        max_neighbors_aint=1000,
        rbf={"name": "gaussian"},
        envelope={"name":"polynomial", "exponent":5},
        cbf={"name": "spherical_harmonics"},
        sbf={"name": "legendre_outer"},
        extensive=True,
        output_init="HeOrthogonal",
        activation="silu",
        scale_file=None,
        quad_interaction=True,
        atom_edge_interaction=True,
        edge_atom_interaction=True,
        atom_interaction=True,
        regress_forces=False,
        num_atom_emb_layers=2,
        num_global_out_layers=2,
        qint_tags=[1,2]
    )   
    return model