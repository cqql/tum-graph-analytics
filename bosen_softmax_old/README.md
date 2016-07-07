# Usage

```sh
In order to run it, a compiled and setup bosen is needed

# Compile program; check
$ make PETUUM_ROOT=<path-to-bosen>
# E.g.
$ make PETUUM_ROOT=/home/username/petuum_test/bosen

# Running example dataset, expects path to file /bosen/machinefiles/localserver
$ python /bosen_softmax/script/_launch.py <path-to-localserver>
# E.g. 
$ python /home/username/petuum_test/bosen/machinefiles/localserver

Note: Both paths can be just saved to some environment variables to safe the typing.

# To stop a task prematurely
$ python /bosen_softmax/script/kill.py <path-to-localserver>

```
