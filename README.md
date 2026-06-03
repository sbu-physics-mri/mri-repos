# MRI Tools
![license](https://img.shields.io/github/license/abdrysdale/mri-tools.svg)

A collection of free and open-source software software tools for use in MRI.
Free is meant as in free beer (gratis) and freedom (libre).

To add a project, add the project url to the `urls.toml` file.

## Table of Contents
- [summary](#summary)
- [stats](#stats)
- [tags](#tags)
	- [mri](#mri)
	- [medical-imaging](#medical-imaging)
	- [deep-learning](#deep-learning)
	- [python](#python)
	- [neuroimaging](#neuroimaging)
	- [pytorch](#pytorch)
	- [machine-learning](#machine-learning)
	- [segmentation](#segmentation)
	- [medical-image-processing](#medical-image-processing)
	- [brain-imaging](#brain-imaging)
	- [quality-control](#quality-control)
	- [mri-images](#mri-images)
	- [convolutional-neural-networks](#convolutional-neural-networks)
	- [diffusion-mri](#diffusion-mri)
	- [image-processing](#image-processing)
	- [medical-image-computing](#medical-image-computing)
	- [itk](#itk)
	- [quality-assurance](#quality-assurance)
	- [fmri](#fmri)
	- [image-reconstruction](#image-reconstruction)
	- [image-registration](#image-registration)
	- [super-resolution](#super-resolution)
	- [simulation](#simulation)
	- [magnetic-resonance-imaging](#magnetic-resonance-imaging)
	- [medical-physics](#medical-physics)
	- [c-plus-plus](#c-plus-plus)
	- [r](#r)
	- [registration](#registration)
	- [computer-vision](#computer-vision)
	- [tensorflow](#tensorflow)
	- [julia](#julia)
	- [brain-connectivity](#brain-connectivity)
	- [neuroscience](#neuroscience)
	- [tractography](#tractography)
	- [fetal](#fetal)
	- [bids](#bids)
	- [fastmri-challenge](#fastmri-challenge)
	- [mri-reconstruction](#mri-reconstruction)
	- [dicom](#dicom)
	- [nifti](#nifti)
	- [qa](#qa)
	- [medical-image-analysis](#medical-image-analysis)
- [languages](#languages)
	- [python](#python)
	- [c++](#c++)
	- [julia](#julia)
	- [jupyter-notebook](#jupyter-notebook)
	- [javascript](#javascript)
	- [c](#c)
	- [r](#r)

## Summary
| Repository | Description | Stars | Forks | Last Updated |
|---|---|---|---|---|
| MONAI | AI Toolkit for Healthcare Imaging | 8234 | 1512 | 2026-06-02 |
| Slicer | Multi-platform, free open source software for visualization and image computing. | 2483 | 738 | 2026-06-03 |
| torchio | Medical imaging processing for AI applications. | 2402 | 264 | 2026-06-02 |
| MedicalZooPytorch | A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation | 1910 | 306 | 2026-05-27 |
| fastMRI | A large-scale dataset of both raw MRI measurements and clinical MRI images. | 1525 | 421 | 2026-06-02 |
| nilearn | Machine learning for NeuroImaging in Python | 1400 | 653 | 2026-06-02 |
| medicaldetectiontoolkit | The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.   | 1355 | 293 | 2026-06-01 |
| deepmedic | Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans | 1061 | 343 | 2026-05-26 |
| SimpleITK | SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages. | 1061 | 228 | 2026-06-02 |
| medicaltorch | A medical imaging framework for Pytorch | 874 | 129 | 2026-05-18 |
| freesurfer | Neuroimaging analysis and visualization suite | 841 | 291 | 2026-06-01 |
| nipype | Workflows and interfaces for neuroimaging packages | 825 | 545 | 2026-06-01 |
| nibabel | Python package to access a cacophony of neuro-imaging file formats | 775 | 283 | 2026-06-02 |
| SynthSeg | Contrast-agnostic segmentation of MRI scans | 561 | 156 | 2026-06-02 |
| brainchop | Brainchop: In-browser 3D MRI rendering and segmentation | 524 | 64 | 2026-05-25 |
| nipy | Neuroimaging in Python FMRI analysis package | 412 | 147 | 2026-06-02 |
| mriqc | Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain | 364 | 139 | 2026-05-30 |
| bart | BART: Toolbox for Computational Magnetic Resonance Imaging | 364 | 174 | 2026-05-09 |
| mriviewer | MRI Viewer is a high performance web tool for advanced 2-D and 3-D medical visualizations. | 363 | 103 | 2026-05-29 |
| mrtrix3 | MRtrix3 provides a set of tools to perform various advanced diffusion MRI analyses, including constrained spherical deconvolution (CSD), probabilistic tractography, track-density imaging, and apparent fibre density | 350 | 198 | 2026-05-04 |
| intensity-normalization | Normalize MR image intensities in Python | 342 | 57 | 2026-04-20 |
| PyMVPA | MultiVariate Pattern Analysis in Python | 324 | 137 | 2026-03-13 |
| direct | Deep learning framework for MRI reconstruction | 303 | 49 | 2026-06-01 |
| TractSeg | Automatic White Matter Bundle Segmentation | 262 | 80 | 2026-05-07 |
| spinalcordtoolbox | Comprehensive and open-source library of analysis tools for MRI of the spinal cord. | 262 | 119 | 2026-06-02 |
| gadgetron | Gadgetron - Medical Image Reconstruction Framework | 259 | 167 | 2026-05-28 |
| nitime | Timeseries analysis for neuroscience data | 259 | 84 | 2026-05-30 |
| pypulseq | Pulseq in Python | 205 | 89 | 2026-06-02 |
| KomaMRI.jl | Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development. | 204 | 40 | 2026-05-28 |
| brainGraph | Graph theory analysis of brain MRI data | 195 | 53 | 2026-06-02 |
| qsiprep | Preprocessing of diffusion MRI | 183 | 63 | 2026-05-29 |
| clinicadl | Framework for the reproducible processing of neuroimaging data with deep learning methods | 181 | 60 | 2026-05-26 |
| smriprep | Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools) | 166 | 47 | 2026-05-29 |
| NiftyMIC | NiftyMIC is a research-focused toolkit for motion correction and volumetric image reconstruction of 2D ultra-fast MRI. | 164 | 39 | 2026-05-28 |
| mritopng | A simple python module to make it easy to batch convert DICOM files to PNG images. | 145 | 52 | 2026-03-25 |
| pydeface | defacing utility for MRI images | 140 | 45 | 2026-05-09 |
| openMorph | Curated list of open-access databases with human structural MRI data | 131 | 37 | 2026-05-29 |
| gif_your_nifti | How to create fancy GIFs from an MRI brain image | 128 | 35 | 2026-05-20 |
| ismrmrd | ISMRM Raw Data Format | 123 | 94 | 2026-05-05 |
| RadQy | RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data. | 121 | 40 | 2026-05-30 |
| niworkflows | Common workflows for MRI (anatomical, functional, diffusion, etc) | 115 | 58 | 2026-05-30 |
| NeSVoR | NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction. | 109 | 26 | 2026-05-28 |
| BraTS-Toolkit | Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript. | 106 | 14 | 2026-05-17 |
| quickNAT_pytorch | PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty | 104 | 34 | 2025-12-03 |
| MRIReco.jl | Julia Package for MRI Reconstruction | 98 | 22 | 2026-03-23 |
| NIfTI.jl | Julia module for reading/writing NIfTI MRI files | 83 | 34 | 2026-05-25 |
| virtual-scanner | An end-to-end hybrid MR simulator/console | 75 | 22 | 2026-04-10 |
| SVRTK | MIRTK based SVR reconstruction | 69 | 9 | 2026-05-21 |
| SIRF | Main repository for the CCP SynerBI software | 69 | 32 | 2026-05-13 |
| QUIT | A set of tools for processing Quantitative MR Images | 63 | 21 | 2026-04-15 |
| tensorflow-mri | A Library of TensorFlow Operators for Computational MRI | 49 | 6 | 2026-04-22 |
| DCEMRI.jl | World's fastest DCE MRI analysis toolkit | 39 | 16 | 2026-01-12 |
| DECAES.jl | DEcomposition and Component Analysis of Exponential Signals (DECAES) - a Julia implementation of the UBC Myelin Water Imaging (MWI) toolbox for computing voxelwise T2-distributions of multi spin-echo MRI images. | 38 | 6 | 2026-04-29 |
| popeye | A population receptive field estimation tool | 35 | 16 | 2026-03-18 |
| gropt | A toolbox for MRI gradient design | 32 | 19 | 2026-05-20 |
| DL-DiReCT | DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation | 32 | 7 | 2026-05-13 |
| ukftractography | None | 31 | 31 | 2025-12-10 |
| hazen | Quality assurance framework for Magnetic Resonance Imaging | 31 | 13 | 2026-06-02 |
| mialsuperresolutiontoolkit | The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework. | 31 | 15 | 2026-04-30 |
| disimpy | Massively parallel Monte Carlo diffusion MR simulator written in Python. | 29 | 9 | 2026-03-28 |
| MriResearchTools.jl | Specialized tools for MRI | 27 | 8 | 2026-06-01 |
| flow4D | Python code for processing 4D flow dicoms and write velocity profiles for CFD simulations. | 26 | 7 | 2026-03-13 |
| nlsam | The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI | 25 | 11 | 2026-02-11 |
| pyCoilGen | Magnetic Field Coil Generator for Python, ported from CoilGen | 23 | 11 | 2026-04-20 |
| dafne | Dafne (Deep Anatomical Federated Network) is a collaborative platform to annotate MRI images and train machine learning models without your data ever leaving your machine. | 20 | 6 | 2026-04-21 |
| MRIgeneralizedBloch.jl | None | 20 | 3 | 2026-04-29 |
| eptlib | EPTlib - An open-source, extensible C++ library of electric properties tomography methods | 18 | 3 | 2026-03-25 |
| sHDR | HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM | 17 | 0 | 2026-01-23 |
| scanhub | ScanHub combines multimodal data acquisition and complex data processing in one cloud platform. | 17 | 3 | 2026-05-28 |
| mrQA | mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance | 14 | 6 | 2026-01-30 |
| ukat | UKRIN Kidney Analysis Toolbox | 14 | 4 | 2026-05-07 |
| PowerGrid | GPU accelerated non-Cartesian magnetic resonance imaging reconstruction toolkit | 14 | 13 | 2026-01-23 |
| MyoQMRI | Quantitative methods for muscle MRI | 12 | 3 | 2026-03-09 |
| CoSimPy | Python electromagnetic cosimulation library | 12 | 4 | 2026-01-01 |
| vespa | Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis | 12 | 6 | 2026-05-03 |
| AFFIRM | A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion | 8 | 1 | 2025-07-27 |
| fetal-IQA | Image quality assessment for fetal MRI | 7 | 0 | 2026-01-21 |
| dwybss | Blind Source Separation of diffusion MRI for free-water elimination and tissue characterization. | 2 | 0 | 2026-01-23 |
| MRISafety.jl | MRI safety checks | 0 | 0 | 2025-01-04 |
| madym_python | Mirror of python wrappers to Madym hosted on Manchester QBI GitLab project | 0 | 0 | 2021-11-22 |
| MRDQED | A Magnetic Resonance Data Quality Evaluation Dashboard | 0 | 1 | 2021-01-31 |
## Stats
- Total repos: 81
- Languages:

| Language | Count |
|---|---|
| python | 46 |
| c++ | 13 |
| julia | 7 |
| jupyter notebook | 5 |
| javascript | 3 |
| c | 3 |
| r | 2 |

- Tags:

| Tag | Count |
|---|---|
| mri | 25 |
| medical-imaging | 17 |
| deep-learning | 16 |
| python | 16 |
| neuroimaging | 11 |
| pytorch | 10 |
| machine-learning | 9 |
| segmentation | 7 |
| medical-image-processing | 7 |
| brain-imaging | 6 |
| quality-control | 5 |
| mri-images | 4 |
| convolutional-neural-networks | 4 |
| diffusion-mri | 4 |
| image-processing | 4 |
| medical-image-computing | 4 |
| itk | 3 |
| quality-assurance | 3 |
| fmri | 3 |
| image-reconstruction | 2 |
| image-registration | 2 |
| super-resolution | 2 |
| simulation | 2 |
| magnetic-resonance-imaging | 2 |
| medical-physics | 2 |
| c-plus-plus | 2 |
| r | 2 |
| registration | 2 |
| computer-vision | 2 |
| tensorflow | 2 |
| julia | 2 |
| brain-connectivity | 2 |
| neuroscience | 2 |
| tractography | 2 |
| fetal | 2 |
| bids | 2 |
| fastmri-challenge | 2 |
| mri-reconstruction | 2 |
| dicom | 2 |
| nifti | 2 |
| qa | 2 |
| medical-image-analysis | 2 |

- Licenses:

| Licence | Count |
|---|---|
| other | 21 |
| mit license | 19 |
| apache license 2.0 | 16 |
| bsd 3-clause "new" or "revised" license | 9 |
| gnu general public license v3.0 | 6 |
| none | 4 |
| gnu affero general public license v3.0 | 2 |
| gnu lesser general public license v3.0 | 2 |
| mozilla public license 2.0 | 2 |




## Tags
### Mri <a name="mri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	291 
>- Issues:	14
>- Watchers:	841
>- Last updated: 2026-06-01

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	139 
>- Issues:	91
>- Watchers:	364
>- Last updated: 2026-05-30

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	174 
>- Issues:	18
>- Watchers:	364
>- Last updated: 2026-05-09

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	119 
>- Issues:	392
>- Watchers:	262
>- Last updated: 2026-06-02

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	89 
>- Issues:	37
>- Watchers:	205
>- Last updated: 2026-06-02

- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	40 
>- Issues:	97
>- Watchers:	204
>- Last updated: 2026-05-28

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	166
>- Last updated: 2026-05-29

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	106
>- Last updated: 2026-05-17

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	31
>- Watchers:	83
>- Last updated: 2026-05-25

- [virtual-scanner](https://github.com/imr-framework/virtual-scanner)
>- An end-to-end hybrid MR simulator/console

>- License: GNU Affero General Public License v3.0
>- Languages: `Jupyter Notebook`
>- Tags: mri
>- Forks:	22 
>- Issues:	15
>- Watchers:	75
>- Last updated: 2026-04-10

- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	5
>- Watchers:	69
>- Last updated: 2026-05-21

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-04-22

- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	32
>- Last updated: 2026-05-13

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	36
>- Watchers:	31
>- Last updated: 2026-06-02

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	8 
>- Issues:	3
>- Watchers:	27
>- Last updated: 2026-06-01

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	23
>- Last updated: 2026-04-20

- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	17
>- Last updated: 2026-01-23

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Medical-Imaging <a name="medical-imaging"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	343 
>- Issues:	23
>- Watchers:	1061
>- Last updated: 2026-05-26

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	129 
>- Issues:	17
>- Watchers:	874
>- Last updated: 2026-05-18

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	49 
>- Issues:	8
>- Watchers:	303
>- Last updated: 2026-06-01

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	106
>- Last updated: 2026-05-17

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	32 
>- Issues:	165
>- Watchers:	69
>- Last updated: 2026-05-13

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	17
>- Last updated: 2026-01-23

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Deep-Learning <a name="deep-learning"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1512 
>- Issues:	473
>- Watchers:	8234
>- Last updated: 2026-06-02

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	343 
>- Issues:	23
>- Watchers:	1061
>- Last updated: 2026-05-26

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	129 
>- Issues:	17
>- Watchers:	874
>- Last updated: 2026-05-18

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	174 
>- Issues:	18
>- Watchers:	364
>- Last updated: 2026-05-09

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	49 
>- Issues:	8
>- Watchers:	303
>- Last updated: 2026-06-01

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	32
>- Last updated: 2026-05-13

- [AFFIRM](https://github.com/allard-shi/affirm)
>- A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion

>- License: MIT License
>- Languages: `Python`
>- Tags: deep-learning, fetus, motion
>- Forks:	1 
>- Issues:	0
>- Watchers:	8
>- Last updated: 2025-07-27

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Python <a name="python"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	129 
>- Issues:	17
>- Watchers:	874
>- Last updated: 2026-05-18

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	119 
>- Issues:	392
>- Watchers:	262
>- Last updated: 2026-06-02

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	89 
>- Issues:	37
>- Watchers:	205
>- Last updated: 2026-06-02

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	52 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-03-25

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-04-22

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	36
>- Watchers:	31
>- Last updated: 2026-06-02

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-02-11

- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Neuroimaging <a name="neuroimaging"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	291 
>- Issues:	14
>- Watchers:	841
>- Last updated: 2026-06-01

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	139 
>- Issues:	91
>- Watchers:	364
>- Last updated: 2026-05-30

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Pytorch <a name="pytorch"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1512 
>- Issues:	473
>- Watchers:	8234
>- Last updated: 2026-06-02

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	129 
>- Issues:	17
>- Watchers:	874
>- Last updated: 2026-05-18

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	49 
>- Issues:	8
>- Watchers:	303
>- Last updated: 2026-06-01

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Machine-Learning <a name="machine-learning"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	343 
>- Issues:	23
>- Watchers:	1061
>- Last updated: 2026-05-26

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	129 
>- Issues:	17
>- Watchers:	874
>- Last updated: 2026-05-18

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	139 
>- Issues:	91
>- Watchers:	364
>- Last updated: 2026-05-30

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-04-22

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-02-11

### Segmentation <a name="segmentation"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	106
>- Last updated: 2026-05-17

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Medical-Image-Processing <a name="medical-image-processing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1512 
>- Issues:	473
>- Watchers:	8234
>- Last updated: 2026-06-02

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	17
>- Last updated: 2026-01-23

### Brain-Imaging <a name="brain-imaging"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Quality-Control <a name="quality-control"></a>
- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	139 
>- Issues:	91
>- Watchers:	364
>- Last updated: 2026-05-30

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Mri-Images <a name="mri-images"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	31
>- Watchers:	83
>- Last updated: 2026-05-25

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	8 
>- Issues:	3
>- Watchers:	27
>- Last updated: 2026-06-01

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Convolutional-Neural-Networks <a name="convolutional-neural-networks"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	343 
>- Issues:	23
>- Watchers:	1061
>- Last updated: 2026-05-26

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Diffusion-Mri <a name="diffusion-mri"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	40 
>- Issues:	97
>- Watchers:	204
>- Last updated: 2026-05-28

- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	63 
>- Issues:	114
>- Watchers:	183
>- Last updated: 2026-05-29

- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	6
>- Watchers:	29
>- Last updated: 2026-03-28

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-02-11

### Image-Processing <a name="image-processing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	166
>- Last updated: 2026-05-29

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	36
>- Watchers:	31
>- Last updated: 2026-06-02

### Medical-Image-Computing <a name="medical-image-computing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1512 
>- Issues:	473
>- Watchers:	8234
>- Last updated: 2026-06-02

- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Itk <a name="itk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

### Quality-Assurance <a name="quality-assurance"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	36
>- Watchers:	31
>- Last updated: 2026-06-02

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Fmri <a name="fmri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	31
>- Watchers:	83
>- Last updated: 2026-05-25

### Image-Reconstruction <a name="image-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	32 
>- Issues:	165
>- Watchers:	69
>- Last updated: 2026-05-13

### Image-Registration <a name="image-registration"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	166
>- Last updated: 2026-05-29

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### Super-Resolution <a name="super-resolution"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

### Simulation <a name="simulation"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	40 
>- Issues:	97
>- Watchers:	204
>- Last updated: 2026-05-28

- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Magnetic-Resonance-Imaging <a name="magnetic-resonance-imaging"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-04-22

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	23
>- Last updated: 2026-04-20

### Medical-Physics <a name="medical-physics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	23
>- Last updated: 2026-04-20

### C-Plus-Plus <a name="c-plus-plus"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### R <a name="r"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Registration <a name="registration"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Computer-Vision <a name="computer-vision"></a>
- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	129 
>- Issues:	17
>- Watchers:	874
>- Last updated: 2026-05-18

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Tensorflow <a name="tensorflow"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-04-22

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Julia <a name="julia"></a>
- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	31
>- Watchers:	83
>- Last updated: 2026-05-25

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Brain-Connectivity <a name="brain-connectivity"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Neuroscience <a name="neuroscience"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Tractography <a name="tractography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Fetal <a name="fetal"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	5
>- Watchers:	69
>- Last updated: 2026-05-21

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

### Bids <a name="bids"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	63 
>- Issues:	114
>- Watchers:	183
>- Last updated: 2026-05-29

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

### Fastmri-Challenge <a name="fastmri-challenge"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	49 
>- Issues:	8
>- Watchers:	303
>- Last updated: 2026-06-01

### Mri-Reconstruction <a name="mri-reconstruction"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	49 
>- Issues:	8
>- Watchers:	303
>- Last updated: 2026-06-01

### Dicom <a name="dicom"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	52 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-03-25

### Nifti <a name="nifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	31
>- Watchers:	83
>- Last updated: 2026-05-25

### Qa <a name="qa"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	36
>- Watchers:	31
>- Last updated: 2026-06-02

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Medical-Image-Analysis <a name="medical-image-analysis"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### 3D-Reconstruction <a name="3d-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### 3D-Visualization <a name="3d-visualization"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### Implicit-Neural-Representation <a name="implicit-neural-representation"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### Nerf <a name="nerf"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### Neural-Network <a name="neural-network"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### Neural-Rendering <a name="neural-rendering"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### Transformers <a name="transformers"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	109
>- Last updated: 2026-05-28

### Neural-Networks <a name="neural-networks"></a>
- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	343 
>- Issues:	23
>- Watchers:	1061
>- Last updated: 2026-05-26

### Quality-Reporter <a name="quality-reporter"></a>
- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	139 
>- Issues:	91
>- Watchers:	364
>- Last updated: 2026-05-30

### Glioblastoma <a name="glioblastoma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	106
>- Last updated: 2026-05-17

### Glioma <a name="glioma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	106
>- Last updated: 2026-05-17

### Cardiac <a name="cardiac"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	40 
>- Issues:	97
>- Watchers:	204
>- Last updated: 2026-05-28

### Diffusion <a name="diffusion"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	40 
>- Issues:	97
>- Watchers:	204
>- Last updated: 2026-05-28

### Gpu-Acceleration <a name="gpu-acceleration"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	40 
>- Issues:	97
>- Watchers:	204
>- Last updated: 2026-05-28

### Magnetic-Field-Solver <a name="magnetic-field-solver"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	23
>- Last updated: 2026-04-20

### Nmr <a name="nmr"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	23
>- Last updated: 2026-04-20

### Physics <a name="physics"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	23
>- Last updated: 2026-04-20

### Fcm <a name="fcm"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

### Harmonization <a name="harmonization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

### Intensity-Normalization <a name="intensity-normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

### Normalization <a name="normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

### Standardization <a name="standardization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

### Whitestripe <a name="whitestripe"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

### Zscore <a name="zscore"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	0
>- Watchers:	342
>- Last updated: 2026-04-20

### Csharp <a name="csharp"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Image-Analysis <a name="image-analysis"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Java <a name="java"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Lua <a name="lua"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Ruby <a name="ruby"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Simpleitk <a name="simpleitk"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Swig <a name="swig"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Tcl <a name="tcl"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	228 
>- Issues:	81
>- Watchers:	1061
>- Last updated: 2026-06-02

### Imaging <a name="imaging"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

### Quality-Metrics <a name="quality-metrics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	40 
>- Issues:	1
>- Watchers:	121
>- Last updated: 2026-05-30

### Image-Segmentation <a name="image-segmentation"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	166
>- Last updated: 2026-05-29

### Structural-Mri <a name="structural-mri"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	166
>- Last updated: 2026-05-29

### Surface-Reconstruction <a name="surface-reconstruction"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	166
>- Last updated: 2026-05-29

### 3D-Segmentation <a name="3d-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

### Frontend-App <a name="frontend-app"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

### Javascript <a name="javascript"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

### Mri-Segmentation <a name="mri-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

### Pyodide <a name="pyodide"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

### Tensorflowjs <a name="tensorflowjs"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

### Three-Js <a name="three-js"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	64 
>- Issues:	8
>- Watchers:	524
>- Last updated: 2026-05-25

### Fetus <a name="fetus"></a>
- [AFFIRM](https://github.com/allard-shi/affirm)
>- A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion

>- License: MIT License
>- Languages: `Python`
>- Tags: deep-learning, fetus, motion
>- Forks:	1 
>- Issues:	0
>- Watchers:	8
>- Last updated: 2025-07-27

### Motion <a name="motion"></a>
- [AFFIRM](https://github.com/allard-shi/affirm)
>- A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion

>- License: MIT License
>- Languages: `Python`
>- Tags: deep-learning, fetus, motion
>- Forks:	1 
>- Issues:	0
>- Watchers:	8
>- Last updated: 2025-07-27

### Ml <a name="ml"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-04-22

### Ai <a name="ai"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Bayesian <a name="bayesian"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Biomarkers <a name="biomarkers"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Neuroanatomy <a name="neuroanatomy"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Uncertainty <a name="uncertainty"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	34 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Analysis <a name="analysis"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Cancer-Imaging-Research <a name="cancer-imaging-research"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Dce-Mri <a name="dce-mri"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Mat-Files <a name="mat-files"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Complex-Networks <a name="complex-networks"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Connectome <a name="connectome"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Connectomics <a name="connectomics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Graph-Theory <a name="graph-theory"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Network-Analysis <a name="network-analysis"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Statistics <a name="statistics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	53 
>- Issues:	11
>- Watchers:	195
>- Last updated: 2026-06-02

### Reconstruction <a name="reconstruction"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	5
>- Watchers:	69
>- Last updated: 2026-05-21

### Retrospecitve <a name="retrospecitve"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	5
>- Watchers:	69
>- Last updated: 2026-05-21

### Slice-To-Volume <a name="slice-to-volume"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	5
>- Watchers:	69
>- Last updated: 2026-05-21

### Denoising-Images <a name="denoising-images"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	63 
>- Issues:	114
>- Watchers:	183
>- Last updated: 2026-05-29

### Distortion-Correction <a name="distortion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	63 
>- Issues:	114
>- Watchers:	183
>- Last updated: 2026-05-29

### Motion-Correction <a name="motion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	63 
>- Issues:	114
>- Watchers:	183
>- Last updated: 2026-05-29

### Pipelines <a name="pipelines"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	63 
>- Issues:	114
>- Watchers:	183
>- Last updated: 2026-05-29

### Brain-Mri <a name="brain-mri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

### Decoding <a name="decoding"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

### Mvpa <a name="mvpa"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	653 
>- Issues:	272
>- Watchers:	1400
>- Last updated: 2026-06-02

### Inverse-Problems <a name="inverse-problems"></a>
- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	49 
>- Issues:	8
>- Watchers:	303
>- Last updated: 2026-06-01

### Afni-Brik-Head <a name="afni-brik-head"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Cifti-2 <a name="cifti-2"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Data-Formats <a name="data-formats"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Ecat <a name="ecat"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Gifti <a name="gifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Minc <a name="minc"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Streamlines <a name="streamlines"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Tck <a name="tck"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### Trk <a name="trk"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	283 
>- Issues:	147
>- Watchers:	775
>- Last updated: 2026-06-02

### 3D-Slicer-Extension <a name="3d-slicer-extension"></a>
- [ukftractography](https://github.com/pnlbwh/ukftractography)
>- None

>- License: Other
>- Languages: `C`
>- Tags: 3d-slicer-extension
>- Forks:	31 
>- Issues:	10
>- Watchers:	31
>- Last updated: 2025-12-10

### Denoising-Algorithm <a name="denoising-algorithm"></a>
- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-02-11

### Fusion <a name="fusion"></a>
- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	17
>- Last updated: 2026-01-23

### Hdr <a name="hdr"></a>
- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	17
>- Last updated: 2026-01-23

### Image <a name="image"></a>
- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	17
>- Last updated: 2026-01-23

### Pet-Mr <a name="pet-mr"></a>
- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	32 
>- Issues:	165
>- Watchers:	69
>- Last updated: 2026-05-13

### Cortical-Thickness <a name="cortical-thickness"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	32
>- Last updated: 2026-05-13

### Morphometry <a name="morphometry"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	32
>- Last updated: 2026-05-13

### Bart-Toolbox <a name="bart-toolbox"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	174 
>- Issues:	18
>- Watchers:	364
>- Last updated: 2026-05-09

### Compressed-Sensing <a name="compressed-sensing"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	174 
>- Issues:	18
>- Watchers:	364
>- Last updated: 2026-05-09

### Computational-Imaging <a name="computational-imaging"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	174 
>- Issues:	18
>- Watchers:	364
>- Last updated: 2026-05-09

### Iterative-Methods <a name="iterative-methods"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	174 
>- Issues:	18
>- Watchers:	364
>- Last updated: 2026-05-09

### Fastmri <a name="fastmri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

### Fastmri-Dataset <a name="fastmri-dataset"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	421 
>- Issues:	18
>- Watchers:	1525
>- Last updated: 2026-06-02

### Healthcare-Imaging <a name="healthcare-imaging"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1512 
>- Issues:	473
>- Watchers:	8234
>- Last updated: 2026-06-02

### Monai <a name="monai"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1512 
>- Issues:	473
>- Watchers:	8234
>- Last updated: 2026-06-02

### Python3 <a name="python3"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1512 
>- Issues:	473
>- Watchers:	8234
>- Last updated: 2026-06-02

### Dicom-Converter <a name="dicom-converter"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	52 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-03-25

### Dicom-Images <a name="dicom-images"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	52 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-03-25

### Medical <a name="medical"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	52 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-03-25

### Medical-Images <a name="medical-images"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	52 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-03-25

### Png <a name="png"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	52 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-03-25

### Brain <a name="brain"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Ismrm <a name="ismrm"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Mr-Image <a name="mr-image"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Mri-Brain <a name="mri-brain"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Niqc <a name="niqc"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### 3D-Printing <a name="3d-printing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### 3D-Slicer <a name="3d-slicer"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Computed-Tomography <a name="computed-tomography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Image-Guided-Therapy <a name="image-guided-therapy"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Kitware <a name="kitware"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### National-Institutes-Of-Health <a name="national-institutes-of-health"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Nih <a name="nih"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Qt <a name="qt"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Tcia-Dac <a name="tcia-dac"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Vtk <a name="vtk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	738 
>- Issues:	671
>- Watchers:	2483
>- Last updated: 2026-06-03

### Freesurfer <a name="freesurfer"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	291 
>- Issues:	14
>- Watchers:	841
>- Last updated: 2026-06-01

### Lcn <a name="lcn"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	291 
>- Issues:	14
>- Watchers:	841
>- Last updated: 2026-06-01

### Fetal-Mri <a name="fetal-mri"></a>
- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Semi-Supervised-Learning <a name="semi-supervised-learning"></a>
- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### 3D-Mask-Rcnn <a name="3d-mask-rcnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### 3D-Models <a name="3d-models"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### 3D-Object-Detection <a name="3d-object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Deep-Neural-Networks <a name="deep-neural-networks"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Detection <a name="detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Mask-Rcnn <a name="mask-rcnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Object-Detection <a name="object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Pytorch-Cnn <a name="pytorch-cnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Pytorch-Deeplearning <a name="pytorch-deeplearning"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Pytorch-Implementation <a name="pytorch-implementation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Retina-Net <a name="retina-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Retina-Unet <a name="retina-unet"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Semantic-Segmentation <a name="semantic-segmentation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### U-Net <a name="u-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	48
>- Watchers:	1355
>- Last updated: 2026-06-01

### Alzheimer-Disease <a name="alzheimer-disease"></a>
- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

### Convolutional-Neural-Network <a name="convolutional-neural-network"></a>
- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	54
>- Watchers:	181
>- Last updated: 2026-05-26

### 3D-Convolutional-Network <a name="3d-convolutional-network"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Brats2018 <a name="brats2018"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Brats2019 <a name="brats2019"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Densenet <a name="densenet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Iseg <a name="iseg"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Iseg-Challenge <a name="iseg-challenge"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Medical-Image-Segmentation <a name="medical-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Mrbrains18 <a name="mrbrains18"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Resnet <a name="resnet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Segmentation-Models <a name="segmentation-models"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Unet <a name="unet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Unet-Image-Segmentation <a name="unet-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1910
>- Last updated: 2026-05-27

### Mri-Phantoms <a name="mri-phantoms"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	36
>- Watchers:	31
>- Last updated: 2026-06-02

### Bids-Apps <a name="bids-apps"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

### Nipype <a name="nipype"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

### Workflow <a name="workflow"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	31
>- Last updated: 2026-04-30

### Cuda <a name="cuda"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	6
>- Watchers:	29
>- Last updated: 2026-03-28

### Gpu-Computing <a name="gpu-computing"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	6
>- Watchers:	29
>- Last updated: 2026-03-28

### Monte-Carlo-Simulation <a name="monte-carlo-simulation"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	6
>- Watchers:	29
>- Last updated: 2026-03-28

### Fitting <a name="fitting"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Mrs <a name="mrs"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Rf-Pulse <a name="rf-pulse"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Spectroscopy <a name="spectroscopy"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Wxpython <a name="wxpython"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Augmentation <a name="augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

### Data-Augmentation <a name="data-augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	264 
>- Issues:	26
>- Watchers:	2402
>- Last updated: 2026-06-02

### Neuroimage <a name="neuroimage"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	119 
>- Issues:	392
>- Watchers:	262
>- Last updated: 2026-06-02

### Spinalcord <a name="spinalcord"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	119 
>- Issues:	392
>- Watchers:	262
>- Last updated: 2026-06-02

### Mri-Sequences <a name="mri-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	89 
>- Issues:	37
>- Watchers:	205
>- Last updated: 2026-06-02

### Pulse-Sequences <a name="pulse-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	89 
>- Issues:	37
>- Watchers:	205
>- Last updated: 2026-06-02

### Pulseq <a name="pulseq"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	89 
>- Issues:	37
>- Watchers:	205
>- Last updated: 2026-06-02

### Big-Data <a name="big-data"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

### Brainweb <a name="brainweb"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

### Data-Science <a name="data-science"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

### Dataflow <a name="dataflow"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

### Dataflow-Programming <a name="dataflow-programming"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01

### Workflow-Engine <a name="workflow-engine"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	545 
>- Issues:	428
>- Watchers:	825
>- Last updated: 2026-06-01



## Languages
### Python <a name="python"></a>
### C++ <a name="c++"></a>
### Julia <a name="julia"></a>
### Jupyter Notebook <a name="jupyter-notebook"></a>
### Javascript <a name="javascript"></a>
### C <a name="c"></a>
### R <a name="r"></a>
### Matlab <a name="matlab"></a>
### Typescript <a name="typescript"></a>
