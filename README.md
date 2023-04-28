# Credit Card Fraud Detection Web Application - XGBoosted Neural Networks
 
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="https://github.com/saplanyanki/DS340-440/blob/main/app/static/assets/BLACKLIST.gif" width="350" title="Blacklist Web App">
  </a>

  <h3 align="center">Data Science Capstone Project @ Penn State</h3>

  <p align="center">
    @yankisaplan - @elifreedman
    <br />
    <a href="https://github.com/saplanyanki/DS340-440/tree/main/sources"><strong>Explore the code »</strong></a>
    <br />
    <br />
    <a href="">View Demo</a>
    ·
    <a href="https://github.com/saplanyanki/DS340-440/issues">Report Bug</a>
    ·
    <a href="https://github.com/saplanyanki/DS340-440/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<img src="https://github.com/saplanyanki/DS340-440/blob/main/app/static/assets/BLACKLIST.gif](https://github.com/saplanyanki/DS340-440/blob/main/im1.png)" width="350" title="Blacklist Web">

Project:
* The Blacklist web application is an innovative tool that uses the power of machine learning to detect and prevent credit card fraud. Our team has developed a custom model that employs various techniques to accurately classify fraudulent activity, providing users with a high level of security and peace of mind. The Blacklist architecture consists of three main components: the model, the backend, and the frontend. 
* The model is a supervised neural network that is enhanced by an XGBoost algorithm, which provides the most relevant features for accurate predictions. The Blacklist also uses the Python Flask module to connect the pre-trained model with the backend, enabling fast and efficient prediction of credit card data. 
* The frontend of the Blacklist is designed to be user-friendly and easy to navigate, making it accessible to users of all technical levels. The seamless integration of the model, backend, and frontend ensures that users receive predictions in just a couple of seconds and have a great experience using the application.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Python][Python]][Python-url]
* [![Learn][Learn]][Learn-url]
* [![Tensorflow][Tensorflow]][Tensorflow-url]
* [![Torch][Torch]][Torch-url]
* [![Flask][Flask]][Flask-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Clone the repository and follow Prerequisites.

### Prerequisites

* Download Python Libraries Listed Below:
  ```sh
  bcrypt==4.0.1
  bootstrap-py==1.0.2
  Flask==2.2.3
  Flask-Bcrypt==1.0.1
  Flask-Login==0.6.2
  Flask-SQLAlchemy==3.0.3
  Flask-WTF==1.1.1
  joblib==1.2.0
  numpy==1.22.0
  pandas==1.5.3
  SQLAlchemy==2.0.6
  tensorflow==2.11.0
  WTForms==3.0.1
  XBNet==1.4.6
  xgboost==1.7.3
  torch
  sklearn
  os
  pickle
  tqdm
  matplotlib
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/saplanyanki/DS340-440.git
   ```
2. Install packages
   ```sh
   pip install everything in the requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

* The process begins when the user navigates to our blacklist website. From there, they are able to either log into their previously created account, or to register for an account if they are a new user. When logging into a previously created account, the sign-in page will refresh if the user credentials are incorrect. If the credentials are correct, the user will be redirected to the prediction page. From the registration page, the user is prompted to fill out their name, email, and password. There are restrictions to the passwords a user may create. The password must be at least eight characters and must contain numbers and special characters for password security.Once on the prediction page, the user is able to upload a comma separated value (csv) file of their credit card statement. Once uploaded, the user is then redirected to the prediction dashboard page.Finally, when on the dashboard, the user is able to see whether or not there were fraudulent transactions on their credit card statement. They are also able to view various graphs pertaining to the level of fraud found on that statement. Lastly, the user is also able to select various models to process their data.

*

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Project Roadmap

- [x] Research
- [x] Dashboard
- [x] Modeling
- [x] Testing

See the [open issues](https://github.com/saplanyanki/DS340-440/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Research Paper 1 (Parent Paper): https://arxiv.org/pdf/2106.05239v3.pdf
* Research Paper 2: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00573-8#Abs1
* Research Paper 3: https://bhooi.github.io/papers/birdnest_sdm16.pdf

@misc{sarkar2021xbnet,
      title={XBNet : An Extremely Boosted Neural Network}, 
      author={Tushar Sarkar},
      year={2021},
      eprint={2106.05239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/saplanyanki/DS340-440?style=for-the-badge
[contributors-url]: https://github.com/saplanyanki/DS340-440/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/saplanyanki/DS340-440?style=for-the-badge
[forks-url]: https://github.com/saplanyanki/DS340-440/network/members
[stars-shield]: https://img.shields.io/github/stars/saplanyanki/DS340-440?style=for-the-badge
[stars-url]: https://github.com/saplanyanki/DS340-440/stargazers
[issues-shield]: https://img.shields.io/github/issues/saplanyanki/DS340-440?style=for-the-badge
[issues-url]: https://github.com/saplanyanki/DS340-440/issues
[license-shield]: https://img.shields.io/github/license/saplanyanki/DS340-440?style=for-the-badge
[license-url]: https://github.com/saplanyanki/DS340-440/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Tensorflow]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org/
[Torch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[Torch-url]: https://pytorch.org/
[Flask]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/2.2.x/
[Learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[Learn-url]: https://scikit-learn.org/stable/






