# Gradient Descent with ps-lite

## Generating test data

```sh
$ ./generate.py -h
```

## Splitting data

For `n` workers the input should be distributed over files `data-0`, `data-1` up
to `data-<n - 1>`. The `split.py` script can that `n`-way split for you.

```sh
$ ./split.py <n> airfoil_self_noise.dat
```

Note that you have to provide the original path to the program nonetheless.

## Running the program

```sh
$ ../ps-lite/tracker/dmlc_local.py -n 2 -s 2 ./gdesc airfoil_self_noise.dat
```
