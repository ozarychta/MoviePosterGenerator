# MoviePosterGenerator
Translating movie posters to another movie genre with [SoloGAN - Multimodal Image-to-Image Translation via a Single Generative Adversarial Network](https://arxiv.org/pdf/2008.01681.pdf)

#### Example results:

Animation -> Horror:

<img src="results\lambda_cyc25_random_style/gen_00116_1.jpg"> 
<img src="results\lambda_cyc25_random_style/gen_00101 (1).jpg"> 

#### Repository structure:
* [soloGAN](soloGAN/) - our implementation of SoloGAN created according to paper [Multimodal Image-to-Image Translation via a Single Generative Adversarial Network](https://arxiv.org/pdf/2008.01681.pdf)
* [soloGAN_unofficial](soloGAN_unofficial/) - unofficial SoloGAN implementation from (https://github.com/limtsekheng/SoloGAN) that we tested (with some changes to adjust to our project) 
* [results](results/) - results of translating movie posters with unofficial SoloGAN implementation with different values of lambda_cyc hyperparameter  (controlling the importance of Cycle consistency loss in training)
