
<h1 align="center">
Codesign of Humanoid Robots for Ergonomic Collaboration with Multiple Humans via Genetic Algorithms and Nonlinear Optimization 
</h1>


<div align="center">

C.Sartore, L.Rapetti, F.Bergonti, S.Dafarra, S.Traversaro, D.Pucci  _"Codesign of Humanoid Robots for Ergonomic Collaboration with Multiple Humans via Genetic Algorithms and Nonlinear Optimization "_
in 2023 IEEE-RAS International Conference on Humanoid Robotics (Humanoids)

</div>

<p align="center">


[![Video](https://github.com/ami-iit/paper_sartore_2023_humanoids_codesign-ga-nl/assets/56030908/3ec1c7fa-d23a-4765-a488-49d5e983082a)](https://github.com/ami-iit/paper_sartore_2023_humanoids_codesign-ga-nl/assets/56030908/6e466fbd-fc45-4b28-93e7-f7467a9b6a28)

</p>

<div align="center">
  IEEE-RAS International Conference on Humanoid Robotics
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a> |
  <a href="https://arxiv.org/abs/2312.07459"><b>Paper</b></a> |
  <a href="https://www.youtube.com/watch?v=MlpmiOGxnlU"><b>Video</b></a>
</div>


### Installation


:warning: The repository depends on [HSL for IPOPT (Coin-HSL)](https://www.hsl.rl.ac.uk/ipopt/), to correctly link the library please substitute [this](./Dockerfile#L108) line of the docker image with the absolute path to the `coinhsl.zip`. In particular for the paper experiments Coin-HSL 2019.05.21 have been used, but also later version should work fine. 

⚠️ This repository depends on [docker](https://docs.docker.com/)


To install the repo on a Linux terminal follow the following steps 

```
git clone https://github.com/ami-iit/paper_sartore_2023_humanoids_codesign-ga-nl.git
cd paper_sartore_2023_humanoids_codesign-ga-nl
docker build --tag sartore2023results . 
```

### Running 
- `GA_with_PyGad.py` : script to launch the bilevel optimization framework; 
- `GA_processOutput.py` : script to process the output of the genetic algorithm; 
- `CompareTorqueSimulation.py`: script to process the simulation output.

### Citing this work
```bibtex
@inproceedings{sartore2023codesign,
  title={Codesign of Humanoid Robots for Ergonomic Collaboration with Multiple Humans via Genetic Algorithms and Nonlinear Optimization},
  author={Sartore, Carlotta and Rapetti, Lorenzo and Bergonti, Fabio and Dafarra, Stefano and Traversaro, Silvio and Pucci, Daniele},
  booktitle={2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids)},
  pages={1--8},
  year={2023},
  organization={IEEE}
  doi={10.1109/Humanoids57100.2023.10375237}
}
```
### Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/CarlottaSartore.png" width="40">](https://github.com/GitHubUserName) | [@CarlottaSartore](https://github.com/CarlottaSartore) |
