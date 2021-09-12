# Improving 3D Pose Estimation of Cheetahs in the Wild
- Upload ground truth used in the experiments and provide link to it here.
- Provide example of how to run code.
- Provide some extra qualitative results.

### Pairwise pseudo-measurement graph
In this work, we decided to only generate PPMs from “intuitive” pairwise terms, i.e. infer the  location  of  a  keypoint  that  makes  kinematic  sense  with regard to the cheetah skeleton. The table below provides the base keypoint and it's corresponding PPMs.

Base keypoint           | Pairwise term 1           |  Pairwise term 2
:-------------------------:|:-------------------------:|:-------------------------:
r\_eye | nose | l\_eye
l\_eye | nose | r\_eye
nose | r\_eye | l\_eye
neck\_base | spine | nose
spine | neck\_base | tail\_base
tail\_base | spine | tail1
tail1 | spine | tail\_base
tail2 | tail1 | tail\_base
l\_shoulder | l\_front\_knee | neck\_base
l\_front\_knee | l\_shoulder | l\_front\_ankle
l\_front\_ankle | l\_front\_knee | l\_shoulder
l\_front\_paw | l\_front\_knee | l\_front\_ankle
r\_shoulder | r\_front\_knee | neck\_base
r\_front\_knee | r\_shoulder | r\_front\_ankle
r\_front\_ankle | r\_front\_knee | r\_shoulder
r\_front\_paw | r\_front\_knee | r\_front\_ankle
l\_hip | l\_back\_knee | tail\_base
l\_back\_knee | l\_hip | l\_back\_ankle
l\_back\_ankle | l\_back\_knee | l\_hip
l\_back\_paw | l\_back\_knee | l\_back\_ankle
r\_hip | r\_back\_knee | tail\_base
r\_back\_knee | r\_hip | r\_back\_ankle
r\_back\_ankle | r\_back\_knee | r\_hip
r\_back\_paw | r\_back\_knee | r\_back\_ankle
