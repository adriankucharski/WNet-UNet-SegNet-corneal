# W-Net, U-Net and SegNet for Corneal Endothelium Image Segmentation
<p>Repository contains implementation of convolutional neural network models: W-Net, U-Net and SegNet based for corneal endothelium image segmentation.<p>

# Terms of use
<p>The source code and image dataset may be used for non-commercial research provided you acknowledge the source by citing the following paper:<p>

<ul>
  <li> Adrian Kucharski, Anna Fabijańska, CNN-watershed: A watershed transform with predicted markers for corneal endothelium image segmentation, Biomedical Signal Processing and Control, Volume 68, 2021, 102805, ISSN 1746-8094, https://doi.org/10.1016/j.bspc.2021.102805</li>
</ul>
  
<pre><code>@article{Kucharski2021,
  doi = {10.1016/j.bspc.2021.102805},
  url = {https://doi.org/10.1016/j.bspc.2021.102805},
  year = {2021},
  month = jul,
  publisher = {Elsevier {BV}},
  volume = {68},
  pages = {102805},
  author = {Adrian Kucharski and Anna Fabija{\'{n}}ska},
  title = {{CNN}-watershed: A watershed transform with predicted markers for corneal endothelium image segmentation},
  journal = {Biomedical Signal Processing and Control}
}</code></pre>
 
## Prerequisites
Code was tested on Windows 10 64-bit with Python 3.8, and TensorFlow 2.3.1.

## Folder tree compatible with default config.ini file.
<pre><code>
main_dir
├── Postprocess
│   └── postprocess.py
├── Predicted_images        # contains predicted images, run predict.py
│   └── Fold_0
│   │   └── UNet
│   │   └── SegNet
│   │   └── WNet
│   └── Fold_1
│   │   └── ...
│   └── Fold_2
│   │   └── ...
│   └── Fold_3
│   │   └── ...
│   └── Fold_4  
│       └── ...
├── Trained_model           # contains trained model and training history
│   └── UNet                
│   └── SegNet                
│   └── WNet              
├── Training_data
│   ├── markers             # markers generated from images from ./gt_all/ images
│   ├── gt_all              # ground truth images from http://bioimlab.dei.unipd.it/Endo%20Aliza%20Data%20Set.htm
│   ├── gt                  # put ground truth images here
│   ├── org                 # put original images here
│   └── field               # put region of interest images here
├── config.ini
├── history_show.py
├── models.py
├── others.py
├── predict.py
├── prepare_dataset.py
├── readme.md
├── predict_from_path.py 
└── training.py
</code></pre>

## How to use
Framework is designed to train and predict images with cross_validation setup. Default numbers of folds is 5 (80% train, 20% predict).

<ol>
<li>Config the config.ini file</li>
<li>Run prepare_dataset.py</li>
<li>Run training.py</li>
<li>Run predict.py</li>
</ol>

## Predict from path
To predict from a different path you can setup <i>predict_from_path.py</i> script. For example, you can set up <b>folds_number</b> to 1 in <b>config.ini</b> to generate training data from a whole dataset (from <i>/Training_data/</i>), then you run <i>prepare_dataset.py</i> and <i>training.py</i> to train networks on new data. What you get are trained networks that you can use in <i>predict_from_path.py</i>. Trained models should be in the <i>/Trained_model/network_name/</i> path.

## Content
<ul>
<li> <b>config.ini</b> main configuration file </li>
<li> <b>history_show.py</b> create plot with value of loss and accuracy, run training.py before </li>
<li> <b>models.py</b> contains implementation of cnn models W-Net, U-Net and SegNet based</li>
<li> <b>others.py</b> io and other functions</li>
</ul>
