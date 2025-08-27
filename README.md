# Tactile_Language_Model
Repo for conversion of tactile images to text explanations. 
Datasets can be located and downloaded within the code. The sizes of the datasets are quite large, so you will need a high performance computer in order to train the models. 


## Dependencies 
This library was developed and tested in Python 3.10.12 and uses the following libraries: 

- numpy
- transformers
- torch
- torchvision
- ollama

Optional 
- sentence-transformers

You wil need to <a href="">install ollama</a>
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

```
## Citation 

```BibTeX
@article{shepherd2025texture,
  title={Texture and Friction Classification: Optical TacTip vs. Vibrational Piezoeletric and Accelerometer Tactile Sensors},
  author={Shepherd, Dexter R and Husbands, Phil and Philippides, Andrew and Johnson, Chris},
  journal={Sensors},
  volume={25},
  number={16},
  pages={4971},
  year={2025},
  publisher={MDPI}
}


```