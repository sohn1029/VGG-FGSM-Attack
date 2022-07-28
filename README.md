# VGG-FGSM-Attack
## ATTACK
The first attack was chosen as an evasion attack, which is one of the most famous attacks. The evasion attack is one of the white box attacks that need to know the error from the model. Among the various attack methods used in the white box attack, the FGSM method was used, and images with perturbations were easily obtained. With this attack, one of the CNN networks, the VGG network, was attacked, and after a long wait, the following results were obtained.  
![vgg_attacked](https://user-images.githubusercontent.com/31722713/181442442-e7bcfbb3-ea7e-4cec-adda-a2fc36901bc1.png)
![vgg_attacked_ex](https://user-images.githubusercontent.com/31722713/181442474-1363dcaa-8335-43a2-9d06-554f67105fef.png)

Epsilon is parameter for how much perturbation is given to the image. The result on the left shows that the accuracy of the network decreases as the epsilon size increases, and the result on the right shows the image that failed to predict and its shape. The FGSM (Fast Gradient Sign Method) used in this attack has a simple structure that finds out the loss from the model and maximizes this loss, and the model used in the article introduced about this attack showed a more vulnerable appearance. If you assign 0.1 as the epsilon value and proceed with the attack, you will be able to lower the performance of the model inconspicuously. The figure below shows the test result of the network that are weaker than VGG network.  
![other_attacked](https://user-images.githubusercontent.com/31722713/181442569-879310f9-4a92-4b6a-b98a-e19d2f823ecb.png)

## DEFENSE
As a defense against this attack, adversarial training was conducted on the model. For one training image, an image effected by perturbation with epsilon of 0.1 was generated, and the normal image and the image effected by perturbation were trained together in the model. Since the model learns two pieces of data including the existing training data and perturbation data, the model sees twice the data per epoch than usual and can get high accuracy with fewer epochs. The result of the same attack as before on the trained model is as follows.  

![vgg_defensed](https://user-images.githubusercontent.com/31722713/181442529-e344ec31-641b-4ded-b3ba-4c2d92bc4af2.png)
![vgg_defensed_ex](https://user-images.githubusercontent.com/31722713/181442559-78895fcf-d2ff-425f-a65e-0d2e61b8f182.png)

The result on the left shows the result of the model that survives strongly even when a perturbation of up to 0.3 is given to the image as an epsilon value, and the result on the right shows the pictures that successfully attacked. The reason the result is less than before is because there are few cases where the attack was successful. This was enough to show how well adversarial training performed.
