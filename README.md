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
	- [medical-image-processing](#medical-image-processing)
	- [segmentation](#segmentation)
	- [brain-imaging](#brain-imaging)
	- [quality-control](#quality-control)
	- [medical-image-computing](#medical-image-computing)
	- [convolutional-neural-networks](#convolutional-neural-networks)
	- [diffusion-mri](#diffusion-mri)
	- [mri-images](#mri-images)
	- [image-processing](#image-processing)
	- [fmri](#fmri)
	- [itk](#itk)
	- [quality-assurance](#quality-assurance)
	- [medical-image-analysis](#medical-image-analysis)
	- [tensorflow](#tensorflow)
	- [image-reconstruction](#image-reconstruction)
	- [image-registration](#image-registration)
	- [super-resolution](#super-resolution)
	- [brain-connectivity](#brain-connectivity)
	- [neuroscience](#neuroscience)
	- [r](#r)
	- [tractography](#tractography)
	- [bids](#bids)
	- [fetal](#fetal)
	- [simulation](#simulation)
	- [magnetic-resonance-imaging](#magnetic-resonance-imaging)
	- [medical-physics](#medical-physics)
	- [fastmri-challenge](#fastmri-challenge)
	- [mri-reconstruction](#mri-reconstruction)
	- [julia](#julia)
	- [nifti](#nifti)
	- [qa](#qa)
	- [dicom](#dicom)
	- [medical-images](#medical-images)
	- [computer-vision](#computer-vision)
	- [c-plus-plus](#c-plus-plus)
	- [registration](#registration)
- [languages](#languages)
	- [python](#python)
	- [c++](#c++)
	- [julia](#julia)
	- [jupyter-notebook](#jupyter-notebook)
	- [c](#c)
	- [javascript](#javascript)
	- [r](#r)

## Summary
| Repository | Description | Stars | Forks | Last Updated |
|---|---|---|---|---|
| MONAI | AI Toolkit for Healthcare Imaging | 7851 | 1423 | 2026-02-15 |
| torchio | Medical imaging processing for AI applications. | 2358 | 255 | 2026-02-13 |
| Slicer | Multi-platform, free open source software for visualization and image computing. | 2321 | 696 | 2026-02-15 |
| MedicalZooPytorch | A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation | 1901 | 306 | 2026-02-14 |
| fastMRI | A large-scale dataset of both raw MRI measurements and clinical MRI images. | 1502 | 418 | 2026-02-13 |
| nilearn | Machine learning for NeuroImaging in Python | 1363 | 644 | 2026-02-14 |
| medicaldetectiontoolkit | The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.   | 1346 | 292 | 2026-02-14 |
| deepmedic | Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans | 1057 | 347 | 2026-02-14 |
| SimpleITK | SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages. | 1039 | 224 | 2026-02-06 |
| medicaltorch | A medical imaging framework for Pytorch | 870 | 128 | 2026-02-13 |
| nipype | Workflows and interfaces for neuroimaging packages | 809 | 540 | 2026-02-12 |
| freesurfer | Neuroimaging analysis and visualization suite | 786 | 280 | 2026-02-13 |
| nibabel | Python package to access a cacophony of neuro-imaging file formats | 762 | 275 | 2026-02-11 |
| SynthSeg | Contrast-agnostic segmentation of MRI scans | 531 | 144 | 2026-02-14 |
| brainchop | Brainchop: In-browser 3D MRI rendering and segmentation | 520 | 61 | 2026-02-07 |
| nipy | Neuroimaging in Python FMRI analysis package | 407 | 146 | 2026-02-03 |
| mriviewer | MRI Viewer is a high performance web tool for advanced 2-D and 3-D medical visualizations. | 360 | 103 | 2026-01-13 |
| bart | BART: Toolbox for Computational Magnetic Resonance Imaging | 356 | 175 | 2026-02-15 |
| mriqc | Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain | 346 | 135 | 2026-02-04 |
| intensity-normalization | Normalize MR image intensities in Python | 339 | 58 | 2026-01-28 |
| mrtrix3 | MRtrix3 provides a set of tools to perform various advanced diffusion MRI analyses, including constrained spherical deconvolution (CSD), probabilistic tractography, track-density imaging, and apparent fibre density | 337 | 193 | 2026-02-12 |
| PyMVPA | MultiVariate Pattern Analysis in Python | 324 | 137 | 2026-01-19 |
| direct | Deep learning framework for MRI reconstruction | 296 | 47 | 2026-02-12 |
| spinalcordtoolbox | Comprehensive and open-source library of analysis tools for MRI of the spinal cord. | 255 | 114 | 2026-02-12 |
| nitime | Timeseries analysis for neuroscience data | 255 | 84 | 2026-02-13 |
| gadgetron | Gadgetron - Medical Image Reconstruction Framework | 253 | 165 | 2026-01-23 |
| TractSeg | Automatic White Matter Bundle Segmentation | 253 | 78 | 2026-02-14 |
| brainGraph | Graph theory analysis of brain MRI data | 191 | 54 | 2025-11-13 |
| pypulseq | Pulseq in Python | 191 | 80 | 2026-02-11 |
| KomaMRI.jl | Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development. | 181 | 33 | 2026-02-14 |
| qsiprep | Preprocessing of diffusion MRI | 177 | 62 | 2026-02-04 |
| clinicadl | Framework for the reproducible processing of neuroimaging data with deep learning methods | 177 | 61 | 2026-02-14 |
| smriprep | Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools) | 162 | 47 | 2026-02-15 |
| NiftyMIC | NiftyMIC is a research-focused toolkit for motion correction and volumetric image reconstruction of 2D ultra-fast MRI. | 159 | 39 | 2026-02-09 |
| mritopng | A simple python module to make it easy to batch convert DICOM files to PNG images. | 146 | 51 | 2025-10-31 |
| pydeface | defacing utility for MRI images | 132 | 43 | 2026-01-22 |
| openMorph | Curated list of open-access databases with human structural MRI data | 129 | 38 | 2025-07-11 |
| gif_your_nifti | How to create fancy GIFs from an MRI brain image | 125 | 35 | 2026-01-29 |
| ismrmrd | ISMRM Raw Data Format | 120 | 95 | 2026-01-27 |
| RadQy | RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data. | 112 | 37 | 2026-02-08 |
| niworkflows | Common workflows for MRI (anatomical, functional, diffusion, etc) | 107 | 56 | 2026-01-19 |
| quickNAT_pytorch | PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty | 104 | 35 | 2025-12-03 |
| NeSVoR | NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction. | 100 | 22 | 2026-02-09 |
| BraTS-Toolkit | Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript. | 99 | 14 | 2026-02-01 |
| MRIReco.jl | Julia Package for MRI Reconstruction | 95 | 23 | 2026-02-14 |
| NIfTI.jl | Julia module for reading/writing NIfTI MRI files | 82 | 35 | 2025-12-27 |
| virtual-scanner | An end-to-end hybrid MR simulator/console | 75 | 21 | 2026-01-23 |
| SIRF | Main repository for the CCP SynerBI software | 68 | 29 | 2026-01-29 |
| SVRTK | MIRTK based SVR reconstruction | 65 | 8 | 2026-02-06 |
| QUIT | A set of tools for processing Quantitative MR Images | 64 | 21 | 2026-01-24 |
| tensorflow-mri | A Library of TensorFlow Operators for Computational MRI | 47 | 6 | 2025-11-25 |
| DCEMRI.jl | World's fastest DCE MRI analysis toolkit | 39 | 16 | 2026-01-12 |
| DECAES.jl | DEcomposition and Component Analysis of Exponential Signals (DECAES) - a Julia implementation of the UBC Myelin Water Imaging (MWI) toolbox for computing voxelwise T2-distributions of multi spin-echo MRI images. | 36 | 6 | 2026-02-12 |
| popeye | A population receptive field estimation tool | 34 | 15 | 2025-08-18 |
| ukftractography | None | 31 | 31 | 2025-12-10 |
| mialsuperresolutiontoolkit | The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework. | 30 | 15 | 2025-12-30 |
| DL-DiReCT | DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation | 30 | 7 | 2026-02-13 |
| hazen | Quality assurance framework for Magnetic Resonance Imaging | 29 | 13 | 2026-01-14 |
| disimpy | Massively parallel Monte Carlo diffusion MR simulator written in Python. | 28 | 9 | 2026-02-12 |
| MriResearchTools.jl | Specialized tools for MRI | 26 | 8 | 2026-01-08 |
| gropt | A toolbox for MRI gradient design | 25 | 16 | 2025-09-24 |
| flow4D | Python code for processing 4D flow dicoms and write velocity profiles for CFD simulations. | 25 | 6 | 2026-01-20 |
| nlsam | The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI | 25 | 11 | 2026-02-11 |
| pyCoilGen | Magnetic Field Coil Generator for Python, ported from CoilGen | 21 | 9 | 2026-02-15 |
| MRIgeneralizedBloch.jl | None | 20 | 3 | 2026-02-04 |
| dafne | Dafne (Deep Anatomical Federated Network) is a collaborative platform to annotate MRI images and train machine learning models without your data ever leaving your machine. | 19 | 6 | 2026-02-10 |
| eptlib | EPTlib - An open-source, extensible C++ library of electric properties tomography methods | 17 | 2 | 2026-01-19 |
| sHDR | HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM | 17 | 0 | 2026-01-23 |
| scanhub | ScanHub combines multimodal data acquisition and complex data processing in one cloud platform. | 16 | 3 | 2026-02-01 |
| PowerGrid | GPU accelerated non-Cartesian magnetic resonance imaging reconstruction toolkit | 14 | 13 | 2026-01-23 |
| mrQA | mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance | 14 | 6 | 2026-01-30 |
| ukat | UKRIN Kidney Analysis Toolbox | 12 | 4 | 2024-10-30 |
| CoSimPy | Python electromagnetic cosimulation library | 12 | 4 | 2026-01-01 |
| MyoQMRI | Quantitative methods for muscle MRI | 12 | 3 | 2025-12-08 |
| vespa | Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis | 11 | 6 | 2025-11-12 |
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
| c++ | 12 |
| julia | 7 |
| jupyter notebook | 5 |
| c | 3 |
| javascript | 3 |
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
| medical-image-processing | 7 |
| segmentation | 7 |
| brain-imaging | 6 |
| quality-control | 5 |
| medical-image-computing | 4 |
| convolutional-neural-networks | 4 |
| diffusion-mri | 4 |
| mri-images | 4 |
| image-processing | 4 |
| fmri | 3 |
| itk | 3 |
| quality-assurance | 3 |
| medical-image-analysis | 2 |
| tensorflow | 2 |
| image-reconstruction | 2 |
| image-registration | 2 |
| super-resolution | 2 |
| brain-connectivity | 2 |
| neuroscience | 2 |
| r | 2 |
| tractography | 2 |
| bids | 2 |
| fetal | 2 |
| simulation | 2 |
| magnetic-resonance-imaging | 2 |
| medical-physics | 2 |
| fastmri-challenge | 2 |
| mri-reconstruction | 2 |
| julia | 2 |
| nifti | 2 |
| qa | 2 |
| dicom | 2 |
| medical-images | 2 |
| computer-vision | 2 |
| c-plus-plus | 2 |
| registration | 2 |

- Licenses:

| Licence | Count |
|---|---|
| other | 21 |
| mit license | 19 |
| apache license 2.0 | 16 |
| bsd 3-clause "new" or "revised" license | 9 |
| gnu general public license v3.0 | 6 |
| none | 4 |
| mozilla public license 2.0 | 2 |
| gnu affero general public license v3.0 | 2 |
| gnu lesser general public license v3.0 | 2 |




## Tags
### Mri <a name="mri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	280 
>- Issues:	24
>- Watchers:	786
>- Last updated: 2026-02-13

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	18
>- Watchers:	356
>- Last updated: 2026-02-15

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	135 
>- Issues:	88
>- Watchers:	346
>- Last updated: 2026-02-04

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	114 
>- Issues:	379
>- Watchers:	255
>- Last updated: 2026-02-12

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	80 
>- Issues:	27
>- Watchers:	191
>- Last updated: 2026-02-11

- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	33 
>- Issues:	102
>- Watchers:	181
>- Last updated: 2026-02-14

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	162
>- Last updated: 2026-02-15

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	99
>- Last updated: 2026-02-01

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

- [virtual-scanner](https://github.com/imr-framework/virtual-scanner)
>- An end-to-end hybrid MR simulator/console

>- License: GNU Affero General Public License v3.0
>- Languages: `Jupyter Notebook`
>- Tags: mri
>- Forks:	21 
>- Issues:	15
>- Watchers:	75
>- Last updated: 2026-01-23

- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	65
>- Last updated: 2026-02-06

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	47
>- Last updated: 2025-11-25

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	30
>- Last updated: 2026-02-13

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	57
>- Watchers:	29
>- Last updated: 2026-01-14

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	8 
>- Issues:	3
>- Watchers:	26
>- Last updated: 2026-01-08

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	9 
>- Issues:	3
>- Watchers:	21
>- Last updated: 2026-02-15

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
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1057
>- Last updated: 2026-02-14

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2026-02-13

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	296
>- Last updated: 2026-02-12

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	99
>- Last updated: 2026-02-01

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	29 
>- Issues:	170
>- Watchers:	68
>- Last updated: 2026-01-29

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
>- Forks:	1423 
>- Issues:	522
>- Watchers:	7851
>- Last updated: 2026-02-15

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1057
>- Last updated: 2026-02-14

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2026-02-13

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	18
>- Watchers:	356
>- Last updated: 2026-02-15

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	296
>- Last updated: 2026-02-12

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	30
>- Last updated: 2026-02-13

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
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2026-02-13

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	114 
>- Issues:	379
>- Watchers:	255
>- Last updated: 2026-02-12

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	80 
>- Issues:	27
>- Watchers:	191
>- Last updated: 2026-02-11

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	47
>- Last updated: 2025-11-25

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	57
>- Watchers:	29
>- Last updated: 2026-01-14

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
>- Watchers:	11
>- Last updated: 2025-11-12

### Neuroimaging <a name="neuroimaging"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	280 
>- Issues:	24
>- Watchers:	786
>- Last updated: 2026-02-13

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	135 
>- Issues:	88
>- Watchers:	346
>- Last updated: 2026-02-04

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

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
>- Forks:	1423 
>- Issues:	522
>- Watchers:	7851
>- Last updated: 2026-02-15

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2026-02-13

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	296
>- Last updated: 2026-02-12

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

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
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1057
>- Last updated: 2026-02-14

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2026-02-13

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	135 
>- Issues:	88
>- Watchers:	346
>- Last updated: 2026-02-04

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
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
>- Watchers:	47
>- Last updated: 2025-11-25

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-02-11

### Medical-Image-Processing <a name="medical-image-processing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1423 
>- Issues:	522
>- Watchers:	7851
>- Last updated: 2026-02-15

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

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

### Segmentation <a name="segmentation"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	99
>- Last updated: 2026-02-01

### Brain-Imaging <a name="brain-imaging"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Quality-Control <a name="quality-control"></a>
- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	135 
>- Issues:	88
>- Watchers:	346
>- Last updated: 2026-02-04

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
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

### Medical-Image-Computing <a name="medical-image-computing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1423 
>- Issues:	522
>- Watchers:	7851
>- Last updated: 2026-02-15

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Convolutional-Neural-Networks <a name="convolutional-neural-networks"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1057
>- Last updated: 2026-02-14

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
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
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	33 
>- Issues:	102
>- Watchers:	181
>- Last updated: 2026-02-14

- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	110
>- Watchers:	177
>- Last updated: 2026-02-04

- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	28
>- Last updated: 2026-02-12

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-02-11

### Mri-Images <a name="mri-images"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	8 
>- Issues:	3
>- Watchers:	26
>- Last updated: 2026-01-08

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Image-Processing <a name="image-processing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	162
>- Last updated: 2026-02-15

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	57
>- Watchers:	29
>- Last updated: 2026-01-14

### Fmri <a name="fmri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

### Itk <a name="itk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Quality-Assurance <a name="quality-assurance"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	57
>- Watchers:	29
>- Last updated: 2026-01-14

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
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Tensorflow <a name="tensorflow"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	47
>- Last updated: 2025-11-25

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Image-Reconstruction <a name="image-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	29 
>- Issues:	170
>- Watchers:	68
>- Last updated: 2026-01-29

### Image-Registration <a name="image-registration"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	162
>- Last updated: 2026-02-15

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### Super-Resolution <a name="super-resolution"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Brain-Connectivity <a name="brain-connectivity"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Neuroscience <a name="neuroscience"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### R <a name="r"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Tractography <a name="tractography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Bids <a name="bids"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	110
>- Watchers:	177
>- Last updated: 2026-02-04

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Fetal <a name="fetal"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	65
>- Last updated: 2026-02-06

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Simulation <a name="simulation"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	33 
>- Issues:	102
>- Watchers:	181
>- Last updated: 2026-02-14

- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Magnetic-Resonance-Imaging <a name="magnetic-resonance-imaging"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	47
>- Last updated: 2025-11-25

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	9 
>- Issues:	3
>- Watchers:	21
>- Last updated: 2026-02-15

### Medical-Physics <a name="medical-physics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	9 
>- Issues:	3
>- Watchers:	21
>- Last updated: 2026-02-15

### Fastmri-Challenge <a name="fastmri-challenge"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	296
>- Last updated: 2026-02-12

### Mri-Reconstruction <a name="mri-reconstruction"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	296
>- Last updated: 2026-02-12

### Julia <a name="julia"></a>
- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Nifti <a name="nifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

### Qa <a name="qa"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	57
>- Watchers:	29
>- Last updated: 2026-01-14

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Dicom <a name="dicom"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Medical-Images <a name="medical-images"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Computer-Vision <a name="computer-vision"></a>
- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2026-02-13

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### C-Plus-Plus <a name="c-plus-plus"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Registration <a name="registration"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### 3D-Mask-Rcnn <a name="3d-mask-rcnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### 3D-Models <a name="3d-models"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### 3D-Object-Detection <a name="3d-object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Deep-Neural-Networks <a name="deep-neural-networks"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Detection <a name="detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Mask-Rcnn <a name="mask-rcnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Object-Detection <a name="object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Pytorch-Cnn <a name="pytorch-cnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Pytorch-Deeplearning <a name="pytorch-deeplearning"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Pytorch-Implementation <a name="pytorch-implementation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Retina-Net <a name="retina-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Retina-Unet <a name="retina-unet"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### Semantic-Segmentation <a name="semantic-segmentation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

### U-Net <a name="u-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1346
>- Last updated: 2026-02-14

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

### 3D-Reconstruction <a name="3d-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### 3D-Visualization <a name="3d-visualization"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### Implicit-Neural-Representation <a name="implicit-neural-representation"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### Nerf <a name="nerf"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### Neural-Network <a name="neural-network"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### Neural-Rendering <a name="neural-rendering"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### Transformers <a name="transformers"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	22 
>- Issues:	7
>- Watchers:	100
>- Last updated: 2026-02-09

### Complex-Networks <a name="complex-networks"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Connectome <a name="connectome"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Connectomics <a name="connectomics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Graph-Theory <a name="graph-theory"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Network-Analysis <a name="network-analysis"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Statistics <a name="statistics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Cuda <a name="cuda"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	28
>- Last updated: 2026-02-12

### Gpu-Computing <a name="gpu-computing"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	28
>- Last updated: 2026-02-12

### Monte-Carlo-Simulation <a name="monte-carlo-simulation"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	28
>- Last updated: 2026-02-12

### Bids-Apps <a name="bids-apps"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Nipype <a name="nipype"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Workflow <a name="workflow"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Cardiac <a name="cardiac"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	33 
>- Issues:	102
>- Watchers:	181
>- Last updated: 2026-02-14

### Diffusion <a name="diffusion"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	33 
>- Issues:	102
>- Watchers:	181
>- Last updated: 2026-02-14

### Gpu-Acceleration <a name="gpu-acceleration"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	33 
>- Issues:	102
>- Watchers:	181
>- Last updated: 2026-02-14

### Reconstruction <a name="reconstruction"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	65
>- Last updated: 2026-02-06

### Retrospecitve <a name="retrospecitve"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	65
>- Last updated: 2026-02-06

### Slice-To-Volume <a name="slice-to-volume"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	65
>- Last updated: 2026-02-06

### Fcm <a name="fcm"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

### Harmonization <a name="harmonization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

### Intensity-Normalization <a name="intensity-normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

### Normalization <a name="normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

### Standardization <a name="standardization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

### Whitestripe <a name="whitestripe"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

### Zscore <a name="zscore"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	339
>- Last updated: 2026-01-28

### Cortical-Thickness <a name="cortical-thickness"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	30
>- Last updated: 2026-02-13

### Morphometry <a name="morphometry"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	30
>- Last updated: 2026-02-13

### 3D-Convolutional-Network <a name="3d-convolutional-network"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Brats2018 <a name="brats2018"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Brats2019 <a name="brats2019"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Densenet <a name="densenet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Iseg <a name="iseg"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Iseg-Challenge <a name="iseg-challenge"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Medical-Image-Segmentation <a name="medical-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Mrbrains18 <a name="mrbrains18"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Resnet <a name="resnet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Segmentation-Models <a name="segmentation-models"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Unet <a name="unet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Unet-Image-Segmentation <a name="unet-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	306 
>- Issues:	21
>- Watchers:	1901
>- Last updated: 2026-02-14

### Big-Data <a name="big-data"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

### Brainweb <a name="brainweb"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

### Data-Science <a name="data-science"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

### Dataflow <a name="dataflow"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

### Dataflow-Programming <a name="dataflow-programming"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

### Workflow-Engine <a name="workflow-engine"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	540 
>- Issues:	436
>- Watchers:	809
>- Last updated: 2026-02-12

### Magnetic-Field-Solver <a name="magnetic-field-solver"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	9 
>- Issues:	3
>- Watchers:	21
>- Last updated: 2026-02-15

### Nmr <a name="nmr"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	9 
>- Issues:	3
>- Watchers:	21
>- Last updated: 2026-02-15

### Physics <a name="physics"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	9 
>- Issues:	3
>- Watchers:	21
>- Last updated: 2026-02-15

### Bart-Toolbox <a name="bart-toolbox"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	18
>- Watchers:	356
>- Last updated: 2026-02-15

### Compressed-Sensing <a name="compressed-sensing"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	18
>- Watchers:	356
>- Last updated: 2026-02-15

### Computational-Imaging <a name="computational-imaging"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	18
>- Watchers:	356
>- Last updated: 2026-02-15

### Iterative-Methods <a name="iterative-methods"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	18
>- Watchers:	356
>- Last updated: 2026-02-15

### Inverse-Problems <a name="inverse-problems"></a>
- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	296
>- Last updated: 2026-02-12

### Mri-Phantoms <a name="mri-phantoms"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	57
>- Watchers:	29
>- Last updated: 2026-01-14

### 3D-Segmentation <a name="3d-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

### Frontend-App <a name="frontend-app"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

### Javascript <a name="javascript"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

### Mri-Segmentation <a name="mri-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

### Pyodide <a name="pyodide"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

### Tensorflowjs <a name="tensorflowjs"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

### Three-Js <a name="three-js"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	7
>- Watchers:	520
>- Last updated: 2026-02-07

### Dicom-Converter <a name="dicom-converter"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Dicom-Images <a name="dicom-images"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Medical <a name="medical"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Png <a name="png"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Mri-Sequences <a name="mri-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	80 
>- Issues:	27
>- Watchers:	191
>- Last updated: 2026-02-11

### Pulse-Sequences <a name="pulse-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	80 
>- Issues:	27
>- Watchers:	191
>- Last updated: 2026-02-11

### Pulseq <a name="pulseq"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	80 
>- Issues:	27
>- Watchers:	191
>- Last updated: 2026-02-11

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

### Glioblastoma <a name="glioblastoma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	99
>- Last updated: 2026-02-01

### Glioma <a name="glioma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	99
>- Last updated: 2026-02-01

### Image-Segmentation <a name="image-segmentation"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	162
>- Last updated: 2026-02-15

### Structural-Mri <a name="structural-mri"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	162
>- Last updated: 2026-02-15

### Surface-Reconstruction <a name="surface-reconstruction"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	83
>- Watchers:	162
>- Last updated: 2026-02-15

### Neuroimage <a name="neuroimage"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	114 
>- Issues:	379
>- Watchers:	255
>- Last updated: 2026-02-12

### Spinalcord <a name="spinalcord"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	114 
>- Issues:	379
>- Watchers:	255
>- Last updated: 2026-02-12

### Imaging <a name="imaging"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

### Quality-Metrics <a name="quality-metrics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	112
>- Last updated: 2026-02-08

### Augmentation <a name="augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

### Data-Augmentation <a name="data-augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

### Medical-Imaging-Datasets <a name="medical-imaging-datasets"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

### Medical-Imaging-With-Deep-Learning <a name="medical-imaging-with-deep-learning"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	32
>- Watchers:	2358
>- Last updated: 2026-02-13

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

### Quality-Reporter <a name="quality-reporter"></a>
- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	135 
>- Issues:	88
>- Watchers:	346
>- Last updated: 2026-02-04

### Neural-Networks <a name="neural-networks"></a>
- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1057
>- Last updated: 2026-02-14

### Denoising-Images <a name="denoising-images"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	110
>- Watchers:	177
>- Last updated: 2026-02-04

### Distortion-Correction <a name="distortion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	110
>- Watchers:	177
>- Last updated: 2026-02-04

### Motion-Correction <a name="motion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	110
>- Watchers:	177
>- Last updated: 2026-02-04

### Pipelines <a name="pipelines"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	110
>- Watchers:	177
>- Last updated: 2026-02-04

### Freesurfer <a name="freesurfer"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	280 
>- Issues:	24
>- Watchers:	786
>- Last updated: 2026-02-13

### Lcn <a name="lcn"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	280 
>- Issues:	24
>- Watchers:	786
>- Last updated: 2026-02-13

### Fastmri <a name="fastmri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

### Fastmri-Dataset <a name="fastmri-dataset"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	418 
>- Issues:	18
>- Watchers:	1502
>- Last updated: 2026-02-13

### Pet-Mr <a name="pet-mr"></a>
- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	29 
>- Issues:	170
>- Watchers:	68
>- Last updated: 2026-01-29

### Healthcare-Imaging <a name="healthcare-imaging"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1423 
>- Issues:	522
>- Watchers:	7851
>- Last updated: 2026-02-15

### Monai <a name="monai"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1423 
>- Issues:	522
>- Watchers:	7851
>- Last updated: 2026-02-15

### Python3 <a name="python3"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1423 
>- Issues:	522
>- Watchers:	7851
>- Last updated: 2026-02-15

### 3D-Printing <a name="3d-printing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### 3D-Slicer <a name="3d-slicer"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Computed-Tomography <a name="computed-tomography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Image-Guided-Therapy <a name="image-guided-therapy"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Kitware <a name="kitware"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### National-Institutes-Of-Health <a name="national-institutes-of-health"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Nih <a name="nih"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Qt <a name="qt"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Tcia-Dac <a name="tcia-dac"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Vtk <a name="vtk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	696 
>- Issues:	656
>- Watchers:	2321
>- Last updated: 2026-02-15

### Csharp <a name="csharp"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Image-Analysis <a name="image-analysis"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Java <a name="java"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Lua <a name="lua"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Ruby <a name="ruby"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Simpleitk <a name="simpleitk"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Swig <a name="swig"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Tcl <a name="tcl"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	224 
>- Issues:	71
>- Watchers:	1039
>- Last updated: 2026-02-06

### Fitting <a name="fitting"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Mrs <a name="mrs"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Rf-Pulse <a name="rf-pulse"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Spectroscopy <a name="spectroscopy"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Wxpython <a name="wxpython"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Alzheimer-Disease <a name="alzheimer-disease"></a>
- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

### Convolutional-Neural-Network <a name="convolutional-neural-network"></a>
- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	61 
>- Issues:	53
>- Watchers:	177
>- Last updated: 2026-02-14

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
>- Watchers:	47
>- Last updated: 2025-11-25

### Afni-Brik-Head <a name="afni-brik-head"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Cifti-2 <a name="cifti-2"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Data-Formats <a name="data-formats"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Ecat <a name="ecat"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Gifti <a name="gifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Minc <a name="minc"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Streamlines <a name="streamlines"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Tck <a name="tck"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Trk <a name="trk"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	275 
>- Issues:	151
>- Watchers:	762
>- Last updated: 2026-02-11

### Ai <a name="ai"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Bayesian <a name="bayesian"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Biomarkers <a name="biomarkers"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Neuroanatomy <a name="neuroanatomy"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Uncertainty <a name="uncertainty"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

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

### Brain-Mri <a name="brain-mri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

### Decoding <a name="decoding"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14

### Mvpa <a name="mvpa"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	644 
>- Issues:	266
>- Watchers:	1363
>- Last updated: 2026-02-14



## Languages
### Python <a name="python"></a>
### C++ <a name="c++"></a>
### Julia <a name="julia"></a>
### Jupyter Notebook <a name="jupyter-notebook"></a>
### C <a name="c"></a>
### Javascript <a name="javascript"></a>
### R <a name="r"></a>
### Matlab <a name="matlab"></a>
### Typescript <a name="typescript"></a>
### Swig <a name="swig"></a>
