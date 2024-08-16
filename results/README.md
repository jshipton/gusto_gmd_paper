This directory contains scripts to generate and plot data presented in
the results section of this paper. Each test is in a separate
directory, with the large data files stored on GitHub LFS. To access
these files you must have GitHub LFS installed, following the
instructions
[here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

To upload data to GitHub LFS for this paper, first install GitHub LFS
following the instructions above, unless you are on Pileus, in which
case it is already installed. Then move the data file to the
corresponding results directory for the test and add to LFS with the
command:

`git lfs track "results/$DIRNAME/$FILENAME"`.

Then add to the git repository as usual:

```
git add results/$DIRNAME/$FILENAME
git add .gitattributes
git commit -m "data for $TESTNAME"
git push
```
