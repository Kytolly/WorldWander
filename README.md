<div align="center">
<h1>
WorldWander: Bridging Egocentric and Exocentric Worlds in Video Generation
</h1>



</div>


## 🎬 Overview
Video diffusion models have recently achieved remarkable progress in realism and controllability. However, achieving seamless video translation across different perspectives, such as first-person (egocentric) and third-person (exocentric), remains underexplored. Bridging these perspectives is crucial for filmmaking, embodied AI, and world models.
Motivated by this, we present <b>WorldWander</b>, an in-context learning framework tailored for translating between egocentric and exocentric worlds in video generation. Building upon advanced video diffusion transformers, WorldWander integrates (i) <i>In-Context Perspective Alignment</i> and (ii) <i>Collaborative Position Encoding </i> to efficiently model cross-view synchronization.
Overall framework is shown below:
![Overall Framework](assets/overall_pipeline.png)


## 🤗 Datasets
To further support our task, we curate <b>EgoExo-8K</b>, a large-scale dataset containing synchronized egocentric–exocentric triplets from both <i>synthetic</i> and <i>real-world</i> scenarios. Details are provided in XXX.


## 🔧 Environment
```
git clone https://github.com/showlab/WorldWander.git
# Installation with the requirement.txt
conda create -n WorldWander python=3.10
conda activate WorldWander
pip install -r requirements.txt
# Installation with environment.yml
conda env create -f environment.yml
conda activate WorldWander
```

## 🔥 Train
```
bash scripts/train_wan2.sh
```

## 🚀 Inference
```
bash scripts/inference_wan2.sh
```

## 🎓 Bibtex
👋 If you find this code helpful for your research, please cite:
```
XXX
```
