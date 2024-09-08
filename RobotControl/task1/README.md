## `rc-2023-24/hw1` 

_Solution for the first homework of robot control course._

### Running the code

To run the code, you need to have all images in the `imgs` subdirectory, inside your current directory (or change path in `main`).

Then, you just use:
```
python3 stitch.py
```

At first, it runs `test_projective_transformation`, 
then runs `manual_stitching` function that produces `task_5_stitched.jpg`, `task_6_stitched.jpg`.

After that, it runs `stitch_subsets`, which creates panoramas for images 
1) `hw11.jpg`, `hw12.jpg`, `hw13.jpg`
and
2) `hw11.jpg`, `hw14.jpg`, `hw13.jpg`. 

Parts of code corresponding to each subtask start with comment 
```
############################### Task X ########################################
```
and end with 
```
###############################################################################
```