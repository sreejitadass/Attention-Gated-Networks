from dataio.transformation.transforms import Transformations

def get_dataset_transformation(arch_type, opts=None):
    transform = Transformations(arch_type)
    if opts is not None:
        transform.initialise(opts)
    transform.print()
    return transform.get_transformation()