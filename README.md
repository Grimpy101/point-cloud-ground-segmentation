# A simple ground segmentation algorithm for (mostly) ALS point clouds

Why? Because other methods I tried were kind of slow and some did not work. This should give a decent result in under 3 minutes on fairly large clouds (100+ million points).


## How to use

- Compile the binary and use `--help` to view parameters. Most should be pretty self-explanatory. Parameter `cell-size` accepts an f64 and defines how big the sampling grid is.
- Prepare an initial cloud (las/laz file), set it as input to the program.
- ???
- Profit!


## Caveats

This uses my external kdtree implementation, which should be in one of my other repositories. The directories with both programs should be next to each other for dependencies to link correctly.

Also, **THIS IS WORK IN PROGRESS!** A lot of stuff may not work yet.