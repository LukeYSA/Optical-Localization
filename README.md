# Optical Localization of Acoustically Levitated Objects

This project is the developpment of a program that uses two cameras to deltermine
    the x, y, z position of a acoustically levitated object.

## Components
- ``localization.py`` is the central program that holds the whole localization program.
    When using localization, run this program.
    - Currently it only takes a path to a input image and try to localize the object in it.
- `` EdgeDetection.py`` is an object that takes care of processing the input grid image and
    make a matrix mapping each pixel to a real life coordinate.
- `` ObjectDetection.py`` is an object that takes care of finding the item that we want to
    localize and getting its central pixel so we can check the real life position using the mapping
    matrix from EdgeDetection.

## Work In Progress
- The ability to use a live stream video and continuously detect the object
- The ability to handle tilted images where the vertical grid lines are not upright, options are:
    - Use more complicated interpolation method.
    - Rotate the image so that the image is upright, save the angle of rotation so all future.
        operations require rotating the image first.
- The ability to output the z coordinate in addition to x and y coordinates, using two cameras.
- Handle the loss of accuracy with the vibration of the object.
- Machine learning algorithm to indentify the object for better object detection and localization.
- More rigorous testings to ensure grid can be constructed in various environments and settings, to 
    ensure object detection always detect the object of interest in various environments and settings.

## Current Limitations
- The grid image is required to be perfectly upright (the vertical lines of the grid has to
    be orthogonal to the bottom edge of the image)
- Only able to output the x, y coordinate, function to output z is still under development.

## Note
- Make sure to change the paths in ``EdgeDetection.py``, ``localization.py``, ``takePicture.py``, 
    ``ObjectDetection.py``, and ``colorTesting.py``
