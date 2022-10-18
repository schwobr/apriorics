
# Mask extraction

## Registration

The first step in the mask extraction process is the registration between the IHC and the H&E images. This is done in 3 steps:
1. Compute a global affine registration using landmarks in the thumbnails of both images.
2. Extract large patches from both images using the previously computed registration to find the right coordinates.
3. Compute affine and warped registration transforms for these patches.

The challenging steps are the first and the third, which are detailed below.


### Global affine registration

To perform global affine registration, we first need to define landmarks that are present in both images for all pairs. When available, we use the centroids of large black dots that can be seen in almost all images (**INSERT EXAMPLE IMAGES**). We also compute the centroid of the tissue mask for both images, using a simple threshold on the grayscale values. This centroid is used as a reference point to pair corresponding black dots from IHC and H&E images, or as a single landmark when these dots are not present in one of the images. Then, depending on the number of available black dots:

* When there is one or none (in which case the tissue centroid is used), only a translation is computed.
* When there are two or more, we first compute a rotation, then a scale (identical for x and y if there are only 2 dots) and finally a translation.
In most cases the rotation and the scale are identity transforms and only the translation is relevant. These transforms are expressed as $3\times 3$ matrices defined as such:

$$
\begin{align}
    R &= 
    \begin{pmatrix} 
        \mathmakebox[2em]{\cos{\theta}} & \mathmakebox[2em]{\sin{\theta}} & \mathmakebox[2em]{0} \\
        -\sin{\theta} & \cos{\theta} & 0 \\
        0 & 0 & 1 \\
    \end{pmatrix}
\\
    S &=
    \begin{pmatrix}
        \mathmakebox[2em]{s_x} & \mathmakebox[2em]{0} & \mathmakebox[2em]{0} \\
        0 & s_y & 0 \\
        0 & 0 & 1
    \end{pmatrix}
\\
    T &=
    \begin{pmatrix}
    \mathmakebox[2em]{0} & \mathmakebox[2em]{0} & \mathmakebox[2em]{t_x} \\
    0 & 0 & t_y \\
    0 & 0 & 1
    \end{pmatrix}
\end{align}
$$

$\theta \in ]-\pi;\pi]$ is the angle of rotation; $s_x$ and $s_y \in [0; +\infty[$ are respectively the horizontal and vertical scale factors; $t_x$ and $t_y \in ]-\infty; +\infty]$ are respectively the horizontal and vertical translation offsets (in pixels). Then, the matrix product of these matrix $A=TSR$ represents the affine transform that successively rotates, scales and translates the image. This matrix can be applied to input coordinates $(x,y)$ to get transformed coordinates $(x', y')$:

$$
\begin{gather}
    \begin{pmatrix}
        x' \\
        y' \\
        1 \\
    \end{pmatrix} 
    =
    A 
    \begin{pmatrix}
        x \\
        y \\
        1 \\
    \end{pmatrix}
\end{gather}
$$