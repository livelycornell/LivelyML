# Lively Machine Learning Models
- We integrate (DVBPR + VGG16)/GAN model to generate the best outfit according to user preference.

## Environment
The code is tested under a Linux desktop with a single GTX-1080 Ti GPU.

Requirements:

- TensorFlow 1.3
- Numpy
- PIL

## Datasets

The four fashion datasets:

- *AmazonFashion (3.3GB)* : 64K users, 234K images, 0.5M actions
- *AmazonWomen (6.2GB)*: 97K users, 347K images, 0.8M actions
- *AmazonMen (2.1GB)*: 34K users, 110K images, 0.2M actions
- *Tradesy (3.4GB)*: 33K users, 326K images, 0.6M actions

## Model Training

**Step 1:** Train DVBPR:

```
cd DVBPR
python main.py
```

The default hyper-parameters are defined in *main.py*, you can change them accordingly. AUC (on validation and test set) is recorded in *DVBPR.log*.

**Step 2:** Train GANs:

```
cd GAN
python main.py --train True
```
The default hyper-parameters are defined in *main.py*, you can change them accordingly. Without '--train True', it will load a trained model and generated images for each category (stroed in folder *samples*).

**Step 3:** Preference Maximization:

```
cd PM
python main.py
```

PM is based on pretrained DVBPR and GAN models. It will randomly pick a user for each category, and show the generated images through the optimization process.
