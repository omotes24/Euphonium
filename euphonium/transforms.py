# transforms.py -- rotations of NCHW image batches and their inverses (identical to experiments/core.py: TF)
#   x.rot90(1, (2, 3)) rotates counter-clockwise when the image is displayed with the row index growing downwards.
TF = {
    'id':     (lambda x: x, lambda x: x),
    'rot90':  (lambda x: x.rot90(1, (2, 3)), lambda x: x.rot90(-1, (2, 3))),
    'rot180': (lambda x: x.rot90(2, (2, 3)), lambda x: x.rot90(-2, (2, 3))),
    'rot270': (lambda x: x.rot90(3, (2, 3)), lambda x: x.rot90(-3, (2, 3))),
}
