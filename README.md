# FLIGHT Mode On: A Feather-Light Network for Low-Light Image Enhancement (UC2+ @ CVPR 2023)

**Mustafa Ozcan, Hamza Ergezer, Mustafa Ayazoglu**

Code will be published soon.


## Test
1. Configure '--directory' to define test dataset,
2. Modify '--save' to change saved model parameters according to training dataset 
3. Run test code.
       
       python test.py

**Abstract:** 
Low-light image enhancement (LLIE) is an ill-posed inverse problem due to the lack of knowledge of the desired image which is obtained under ideal illumination conditions. Low-light conditions give rise to two main issues: a suppressed image histogram and inconsistent relative color distributions with low signal-to-noise ratio. In order to address these problems, we propose a novel approach named FLIGHT-Net using a sequence of neural architecture blocks. The first block regulates illumination conditions through pixel-wise scene dependent illumination adjustment. The output image is produced in the output of the second block, which includes channel attention and denoising sub-blocks. Our highly efficient neural network architecture delivers state-of-the-art performance with only 25K parameters. The method's code, pretrained models and resulting images will be publicly available


[\[Preprint\]](https://arxiv.org/abs/2305.10889)
 [\[CVF Open Access\]](https://openaccess.thecvf.com/content/CVPR2023W/UG2/html/Ozcan_FLIGHT_Mode_On_A_Feather-Light_Network_for_Low-Light_Image_Enhancement_CVPRW_2023_paper.html)

## Citation

    @InProceedings{Ozcan_2023_CVPR,
        author    = {Ozcan, Mustafa and Ergezer, Hamza and Ayazoglu, Mustafa},
        title     = {FLIGHT Mode On: A Feather-Light Network for Low-Light Image Enhancement},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2023},
        pages     = {4225-4234}
    }

## License and Acknowledgement

This work is under CC BY-NC-SA 4.0 license.
