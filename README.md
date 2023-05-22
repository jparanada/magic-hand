# Magic Hand

Your favorite Rocket Gang Secret Mecha.

## Requirements

You will need the following software installed on your personal computer:

* `scanner_refl_fix` - https://github.com/doug3236/scanner_refl_fix
  * referred to as "srf" throughout
  * have fun compiling this from source on a non-Windows machine
* ArgyllCMS - https://argyllcms.com/
* `oxipng` - https://github.com/shssoichiro/oxipng

You will also need some color profiles. For working spaces we're using 
https://github.com/ellelstone/elles_icc_profiles/tree/master/profiles because they are 
[well-behaved](https://ninedegreesbelow.com/photography/well-behaved-profile.html) and it's easy to see how these were 
created. You will need a scanner profile (aka input profile), which is particular to your device (and which was 
generated with a target image that was first processed with srf prior to creating the profile). You will need the sRGB 
profile, which is included in Elle's profiles as well as ArgyllCMS; however, I am choosing to use the sRGB that is 
included on Windows/Mac by default, since that is what most other folks have (even though it's not well-behaved).

Lastly, you will need a checkerboard calibration grid. For < 1 pixel of error on the features, scanning at 1600 dpi, 
you likely would want something with < 15.875 μm tolerance. I used the 
[WMT-127-3.0-C](https://www.dot-vision.com/Product/Checkerboard-Series-chrome-on-ceramic.html) grid from Dot Vision.

## Instructions

Just press the scan button on your scanner and it makes the picture, idk what you need all this stuff for. 🙃

IT IS VERY IMPORTANT THAT THE INPUT TIFF BE TAGGED WITH ITS CORRECT DPI, since srf needs to know the 
physical size of the image to work! Epson Scan will tag the image correctly, and image editors like Affinity Photo &
Photoshop will preserve any DPI that's been set, so as long as you didn't do something like bring the scan into a
project with different DPI, you should be fine here.

(TODO)

## Suggested Substitutes

* If you don't have a calibration grid, you'd skip the camera calibration step entirely. When centering you'd use the
same edge and corner-detection logic, only the destination points for the corners would be four points that you choose,
based on the era of card you are scanning.
* srf can be omitted.

## Troubleshooting

### _TIFFVSetField: <some_file>.tif: Bad value 32764 for "ResolutionUnit" tag.

This means libtiff found some metadata it didn't understand; however, it will just get ignored. This won't affect the
files you're generating in any way and you can just ignore this warning.
