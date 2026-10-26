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
	- [python](#python)
	- [deep-learning](#deep-learning)
	- [neuroimaging](#neuroimaging)
	- [pytorch](#pytorch)
	- [machine-learning](#machine-learning)
	- [segmentation](#segmentation)
	- [medical-image-processing](#medical-image-processing)
	- [brain-imaging](#brain-imaging)
	- [quality-control](#quality-control)
	- [image-processing](#image-processing)
	- [medical-image-computing](#medical-image-computing)
	- [mri-images](#mri-images)
	- [diffusion-mri](#diffusion-mri)
	- [convolutional-neural-networks](#convolutional-neural-networks)
	- [quality-assurance](#quality-assurance)
	- [itk](#itk)
	- [fmri](#fmri)
	- [dicom](#dicom)
	- [nifti](#nifti)
	- [qa](#qa)
	- [c-plus-plus](#c-plus-plus)
	- [registration](#registration)
	- [tractography](#tractography)
	- [julia](#julia)
	- [r](#r)
	- [fetal](#fetal)
	- [neuroscience](#neuroscience)
	- [image-registration](#image-registration)
	- [image-reconstruction](#image-reconstruction)
	- [bids](#bids)
	- [magnetic-resonance-imaging](#magnetic-resonance-imaging)
	- [medical-physics](#medical-physics)
	- [simulation](#simulation)
	- [super-resolution](#super-resolution)
	- [brain-connectivity](#brain-connectivity)
	- [fastmri-challenge](#fastmri-challenge)
	- [mri-reconstruction](#mri-reconstruction)
	- [medical-image-analysis](#medical-image-analysis)
	- [tensorflow](#tensorflow)
	- [computer-vision](#computer-vision)
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
| MONAI | AI Toolkit for Healthcare Imaging | 8804 | 1685 | 2026-10-26 |
| Slicer | Multi-platform, free open source software for visualization and image computing. | 2681 | 798 | 2026-10-25 |
| torchio | Medical imaging processing for AI applications. | 2450 | 288 | 2026-10-23 |
| MedicalZooPytorch | A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation | 1921 | 305 | 2026-10-15 |
| fastMRI | A large-scale dataset of both raw MRI measurements and clinical MRI images. | 1537 | 422 | 2026-10-25 |
| nilearn | Machine learning for NeuroImaging in Python | 1452 | 691 | 2026-10-25 |
| medicaldetectiontoolkit | The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.   | 1362 | 293 | 2026-10-25 |
| SimpleITK | SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages. | 1092 | 236 | 2026-10-23 |
| deepmedic | Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans | 1068 | 341 | 2026-10-22 |
| freesurfer | Neuroimaging analysis and visualization suite | 881 | 303 | 2026-10-26 |
| medicaltorch | A medical imaging framework for Pytorch | 872 | 130 | 2026-08-21 |
| nipype | Workflows and interfaces for neuroimaging packages | 835 | 541 | 2026-10-14 |
| nibabel | Python package to access a cacophony of neuro-imaging file formats | 798 | 308 | 2026-10-26 |
| SynthSeg | Contrast-agnostic segmentation of MRI scans | 587 | 163 | 2026-10-24 |
| brainchop | Brainchop: In-browser 3D MRI rendering and segmentation | 577 | 65 | 2026-10-26 |
| nipy | Neuroimaging in Python FMRI analysis package | 419 | 147 | 2026-10-21 |
| mriqc | Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain | 381 | 142 | 2026-10-14 |
| bart | BART: Toolbox for Computational Magnetic Resonance Imaging | 370 | 177 | 2026-07-27 |
| mriviewer | MRI Viewer is a high performance web tool for advanced 2-D and 3-D medical visualizations. | 370 | 105 | 2026-10-25 |
| mrtrix3 | MRtrix3 provides a set of tools to perform various advanced diffusion MRI analyses, including constrained spherical deconvolution (CSD), probabilistic tractography, track-density imaging, and apparent fibre density | 356 | 206 | 2026-10-21 |
| intensity-normalization | Normalize MR image intensities in Python | 340 | 57 | 2026-08-17 |
| PyMVPA | MultiVariate Pattern Analysis in Python | 325 | 135 | 2026-10-25 |
| direct | Deep learning framework for MRI reconstruction | 320 | 50 | 2026-08-29 |
| spinalcordtoolbox | Comprehensive and open-source library of analysis tools for MRI of the spinal cord. | 274 | 123 | 2026-10-19 |
| TractSeg | Automatic White Matter Bundle Segmentation | 270 | 82 | 2026-10-14 |
| gadgetron | Gadgetron - Medical Image Reconstruction Framework | 264 | 165 | 2026-10-12 |
| nitime | Timeseries analysis for neuroscience data | 264 | 84 | 2026-10-22 |
| KomaMRI.jl | Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development. | 225 | 55 | 2026-10-19 |
| pypulseq | Pulseq in Python | 221 | 99 | 2026-10-20 |
| brainGraph | Graph theory analysis of brain MRI data | 200 | 52 | 2026-10-12 |
| qsiprep | Preprocessing of diffusion MRI | 195 | 64 | 2026-10-25 |
| clinicadl | Open-source Python library for reproducible deep learning in neuroimaging | 191 | 66 | 2026-10-20 |
| smriprep | Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools) | 169 | 48 | 2026-10-06 |
| NiftyMIC | NiftyMIC is a research-focused toolkit for motion correction and volumetric image reconstruction of 2D ultra-fast MRI. | 167 | 38 | 2026-09-24 |
| mritopng | A simple python module to make it easy to batch convert DICOM files to PNG images. | 145 | 53 | 2026-09-21 |
| pydeface | defacing utility for MRI images | 142 | 47 | 2026-09-24 |
| openMorph | Curated list of open-access databases with human structural MRI data | 131 | 37 | 2026-08-20 |
| gif_your_nifti | How to create fancy GIFs from an MRI brain image | 130 | 35 | 2026-10-09 |
| RadQy | RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data. | 126 | 41 | 2026-10-22 |
| ismrmrd | ISMRM Raw Data Format | 122 | 94 | 2026-09-11 |
| niworkflows | Common workflows for MRI (anatomical, functional, diffusion, etc) | 121 | 59 | 2026-10-25 |
| NeSVoR | NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction. | 117 | 26 | 2026-10-01 |
| BraTS-Toolkit | Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript. | 108 | 14 | 2026-10-20 |
| quickNAT_pytorch | PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty | 104 | 35 | 2025-12-03 |
| MRIReco.jl | Julia Package for MRI Reconstruction | 97 | 26 | 2026-08-17 |
| NIfTI.jl | Julia module for reading/writing NIfTI MRI files | 83 | 34 | 2026-09-15 |
| SVRTK | MIRTK based SVR reconstruction | 79 | 9 | 2026-10-23 |
| SIRF | Main repository for the CCP SynerBI software | 77 | 32 | 2026-10-15 |
| virtual-scanner | An end-to-end hybrid MR simulator/console | 76 | 21 | 2026-07-18 |
| QUIT | A set of tools for processing Quantitative MR Images | 64 | 21 | 2026-08-06 |
| tensorflow-mri | A Library of TensorFlow Operators for Computational MRI | 49 | 6 | 2026-08-14 |
| DCEMRI.jl | World's fastest DCE MRI analysis toolkit | 41 | 15 | 2026-09-30 |
| DECAES.jl | DEcomposition and Component Analysis of Exponential Signals (DECAES) - a Julia implementation of the UBC Myelin Water Imaging (MWI) toolbox for computing voxelwise T2-distributions of multi spin-echo MRI images. | 40 | 6 | 2026-10-18 |
| gropt | A toolbox for MRI gradient design | 38 | 20 | 2026-08-18 |
| popeye | A population receptive field estimation tool | 35 | 16 | 2026-03-18 |
| hazen | Quality assurance framework for Magnetic Resonance Imaging | 35 | 15 | 2026-10-08 |
| flow4D | Python code for processing 4D flow dicoms and write velocity profiles for CFD simulations. | 35 | 6 | 2026-10-21 |
| DL-DiReCT | DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation | 34 | 7 | 2026-08-06 |
| mialsuperresolutiontoolkit | The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework. | 33 | 15 | 2026-07-24 |
| ukftractography | None | 30 | 31 | 2026-08-14 |
| disimpy | Massively parallel Monte Carlo diffusion MR simulator written in Python. | 30 | 9 | 2026-06-16 |
| pyCoilGen | Magnetic Field Coil Generator for Python, ported from CoilGen | 27 | 11 | 2026-09-22 |
| MriResearchTools.jl | Specialized tools for MRI | 27 | 9 | 2026-09-02 |
| nlsam | The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI | 25 | 11 | 2026-09-09 |
| dafne | Dafne (Deep Anatomical Federated Network) is a collaborative platform to annotate MRI images and train machine learning models without your data ever leaving your machine. | 22 | 6 | 2026-09-28 |
| MRIgeneralizedBloch.jl | None | 21 | 3 | 2026-08-18 |
| scanhub | ScanHub combines multimodal data acquisition and complex data processing in one cloud platform. | 19 | 3 | 2026-10-16 |
| eptlib | EPTlib - An open-source, extensible C++ library of electric properties tomography methods | 19 | 3 | 2026-09-12 |
| sHDR | HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM | 17 | 0 | 2026-01-23 |
| PowerGrid | GPU accelerated non-Cartesian magnetic resonance imaging reconstruction toolkit | 14 | 13 | 2026-01-23 |
| mrQA | mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance | 14 | 6 | 2026-01-30 |
| ukat | UKRIN Kidney Analysis Toolbox | 14 | 4 | 2026-05-07 |
| MyoQMRI | Quantitative methods for muscle MRI | 14 | 4 | 2026-10-20 |
| CoSimPy | Python electromagnetic cosimulation library | 13 | 4 | 2026-08-05 |
| vespa | Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis | 12 | 6 | 2026-05-03 |
| AFFIRM | A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion | 8 | 1 | 2025-07-27 |
| fetal-IQA | Image quality assessment for fetal MRI | 7 | 0 | 2026-01-21 |
| dwybss | Blind Source Separation of diffusion MRI for free-water elimination and tissue characterization. | 2 | 0 | 2026-01-23 |
| MRDQED | A Magnetic Resonance Data Quality Evaluation Dashboard | 0 | 1 | 2021-01-31 |
| MRISafety.jl | MRI safety checks | 0 | 0 | 2025-01-04 |
| madym_python | Mirror of python wrappers to Madym hosted on Manchester QBI GitLab project | 0 | 0 | 2021-11-22 |
## Stats
- Total repos: 81
- Languages:

| Language | Count |
|---|---|
| python | 46 |
| c++ | 13 |
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
| python | 16 |
| deep-learning | 16 |
| neuroimaging | 11 |
| pytorch | 10 |
| machine-learning | 9 |
| segmentation | 7 |
| medical-image-processing | 7 |
| brain-imaging | 6 |
| quality-control | 5 |
| image-processing | 4 |
| medical-image-computing | 4 |
| mri-images | 4 |
| diffusion-mri | 4 |
| convolutional-neural-networks | 4 |
| quality-assurance | 3 |
| itk | 3 |
| fmri | 3 |
| dicom | 2 |
| nifti | 2 |
| qa | 2 |
| c-plus-plus | 2 |
| registration | 2 |
| tractography | 2 |
| julia | 2 |
| r | 2 |
| fetal | 2 |
| neuroscience | 2 |
| image-registration | 2 |
| image-reconstruction | 2 |
| bids | 2 |
| magnetic-resonance-imaging | 2 |
| medical-physics | 2 |
| simulation | 2 |
| super-resolution | 2 |
| brain-connectivity | 2 |
| fastmri-challenge | 2 |
| mri-reconstruction | 2 |
| medical-image-analysis | 2 |
| tensorflow | 2 |
| computer-vision | 2 |

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
| mozilla public license 2.0 | 2 |
| gnu lesser general public license v3.0 | 2 |




## Tags
### Mri <a name="mri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	303 
>- Issues:	27
>- Watchers:	881
>- Last updated: 2026-10-26

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	142 
>- Issues:	101
>- Watchers:	381
>- Last updated: 2026-10-14

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	177 
>- Issues:	18
>- Watchers:	370
>- Last updated: 2026-07-27

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	123 
>- Issues:	431
>- Watchers:	274
>- Last updated: 2026-10-19

- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	55 
>- Issues:	89
>- Watchers:	225
>- Last updated: 2026-10-19

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	99 
>- Issues:	38
>- Watchers:	221
>- Last updated: 2026-10-20

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	48 
>- Issues:	84
>- Watchers:	169
>- Last updated: 2026-10-06

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	108
>- Last updated: 2026-10-20

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	30
>- Watchers:	83
>- Last updated: 2026-09-15

- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	7
>- Watchers:	79
>- Last updated: 2026-10-23

- [virtual-scanner](https://github.com/imr-framework/virtual-scanner)
>- An end-to-end hybrid MR simulator/console

>- License: GNU Affero General Public License v3.0
>- Languages: `Jupyter Notebook`
>- Tags: mri
>- Forks:	21 
>- Issues:	15
>- Watchers:	76
>- Last updated: 2026-07-18

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-08-14

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	15 
>- Issues:	22
>- Watchers:	35
>- Last updated: 2026-10-08

- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	34
>- Last updated: 2026-08-06

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	27
>- Last updated: 2026-09-22

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	9 
>- Issues:	5
>- Watchers:	27
>- Last updated: 2026-09-02

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
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	341 
>- Issues:	23
>- Watchers:	1068
>- Last updated: 2026-10-22

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	130 
>- Issues:	17
>- Watchers:	872
>- Last updated: 2026-08-21

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	50 
>- Issues:	6
>- Watchers:	320
>- Last updated: 2026-08-29

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Open-source Python library for reproducible deep learning in neuroimaging

>- License: MIT License
>- Languages: `Python`
>- Tags: brain-imaging, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	66 
>- Issues:	42
>- Watchers:	191
>- Last updated: 2026-10-20

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	108
>- Last updated: 2026-10-20

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	32 
>- Issues:	157
>- Watchers:	77
>- Last updated: 2026-10-15

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	15 
>- Issues:	5
>- Watchers:	41
>- Last updated: 2026-09-30

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

### Python <a name="python"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	130 
>- Issues:	17
>- Watchers:	872
>- Last updated: 2026-08-21

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	123 
>- Issues:	431
>- Watchers:	274
>- Last updated: 2026-10-19

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	99 
>- Issues:	38
>- Watchers:	221
>- Last updated: 2026-10-20

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Open-source Python library for reproducible deep learning in neuroimaging

>- License: MIT License
>- Languages: `Python`
>- Tags: brain-imaging, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	66 
>- Issues:	42
>- Watchers:	191
>- Last updated: 2026-10-20

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	53 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-09-21

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-08-14

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	15 
>- Issues:	22
>- Watchers:	35
>- Last updated: 2026-10-08

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-09-09

- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Deep-Learning <a name="deep-learning"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1685 
>- Issues:	522
>- Watchers:	8804
>- Last updated: 2026-10-26

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	341 
>- Issues:	23
>- Watchers:	1068
>- Last updated: 2026-10-22

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	130 
>- Issues:	17
>- Watchers:	872
>- Last updated: 2026-08-21

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	177 
>- Issues:	18
>- Watchers:	370
>- Last updated: 2026-07-27

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	50 
>- Issues:	6
>- Watchers:	320
>- Last updated: 2026-08-29

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Open-source Python library for reproducible deep learning in neuroimaging

>- License: MIT License
>- Languages: `Python`
>- Tags: brain-imaging, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	66 
>- Issues:	42
>- Watchers:	191
>- Last updated: 2026-10-20

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
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
>- Watchers:	34
>- Last updated: 2026-08-06

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

### Neuroimaging <a name="neuroimaging"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	303 
>- Issues:	27
>- Watchers:	881
>- Last updated: 2026-10-26

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	142 
>- Issues:	101
>- Watchers:	381
>- Last updated: 2026-10-14

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Open-source Python library for reproducible deep learning in neuroimaging

>- License: MIT License
>- Languages: `Python`
>- Tags: brain-imaging, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	66 
>- Issues:	42
>- Watchers:	191
>- Last updated: 2026-10-20

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
>- Forks:	1685 
>- Issues:	522
>- Watchers:	8804
>- Last updated: 2026-10-26

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	130 
>- Issues:	17
>- Watchers:	872
>- Last updated: 2026-08-21

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	50 
>- Issues:	6
>- Watchers:	320
>- Last updated: 2026-08-29

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Open-source Python library for reproducible deep learning in neuroimaging

>- License: MIT License
>- Languages: `Python`
>- Tags: brain-imaging, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	66 
>- Issues:	42
>- Watchers:	191
>- Last updated: 2026-10-20

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

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

### Machine-Learning <a name="machine-learning"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	341 
>- Issues:	23
>- Watchers:	1068
>- Last updated: 2026-10-22

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	130 
>- Issues:	17
>- Watchers:	872
>- Last updated: 2026-08-21

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	142 
>- Issues:	101
>- Watchers:	381
>- Last updated: 2026-10-14

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

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
>- Watchers:	49
>- Last updated: 2026-08-14

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-09-09

### Segmentation <a name="segmentation"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	108
>- Last updated: 2026-10-20

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Medical-Image-Processing <a name="medical-image-processing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1685 
>- Issues:	522
>- Watchers:	8804
>- Last updated: 2026-10-26

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	15 
>- Issues:	5
>- Watchers:	41
>- Last updated: 2026-09-30

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
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Open-source Python library for reproducible deep learning in neuroimaging

>- License: MIT License
>- Languages: `Python`
>- Tags: brain-imaging, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	66 
>- Issues:	42
>- Watchers:	191
>- Last updated: 2026-10-20

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
>- Forks:	142 
>- Issues:	101
>- Watchers:	381
>- Last updated: 2026-10-14

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

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

### Image-Processing <a name="image-processing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	48 
>- Issues:	84
>- Watchers:	169
>- Last updated: 2026-10-06

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	15 
>- Issues:	22
>- Watchers:	35
>- Last updated: 2026-10-08

### Medical-Image-Computing <a name="medical-image-computing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1685 
>- Issues:	522
>- Watchers:	8804
>- Last updated: 2026-10-26

- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

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
>- Forks:	34 
>- Issues:	30
>- Watchers:	83
>- Last updated: 2026-09-15

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	9 
>- Issues:	5
>- Watchers:	27
>- Last updated: 2026-09-02

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Diffusion-Mri <a name="diffusion-mri"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	55 
>- Issues:	89
>- Watchers:	225
>- Last updated: 2026-10-19

- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	64 
>- Issues:	80
>- Watchers:	195
>- Last updated: 2026-10-25

- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	7
>- Watchers:	30
>- Last updated: 2026-06-16

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-09-09

### Convolutional-Neural-Networks <a name="convolutional-neural-networks"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	341 
>- Issues:	23
>- Watchers:	1068
>- Last updated: 2026-10-22

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

### Quality-Assurance <a name="quality-assurance"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	15 
>- Issues:	22
>- Watchers:	35
>- Last updated: 2026-10-08

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Itk <a name="itk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

### Fmri <a name="fmri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	30
>- Watchers:	83
>- Last updated: 2026-09-15

### Dicom <a name="dicom"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	53 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-09-21

### Nifti <a name="nifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	30
>- Watchers:	83
>- Last updated: 2026-09-15

### Qa <a name="qa"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	15 
>- Issues:	22
>- Watchers:	35
>- Last updated: 2026-10-08

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### C-Plus-Plus <a name="c-plus-plus"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Registration <a name="registration"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Tractography <a name="tractography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Julia <a name="julia"></a>
- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	34 
>- Issues:	30
>- Watchers:	83
>- Last updated: 2026-09-15

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	15 
>- Issues:	5
>- Watchers:	41
>- Last updated: 2026-09-30

### R <a name="r"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Fetal <a name="fetal"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	7
>- Watchers:	79
>- Last updated: 2026-10-23

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

### Neuroscience <a name="neuroscience"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	14
>- Last updated: 2026-01-30

### Image-Registration <a name="image-registration"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	48 
>- Issues:	84
>- Watchers:	169
>- Last updated: 2026-10-06

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### Image-Reconstruction <a name="image-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	32 
>- Issues:	157
>- Watchers:	77
>- Last updated: 2026-10-15

### Bids <a name="bids"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	64 
>- Issues:	80
>- Watchers:	195
>- Last updated: 2026-10-25

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

### Magnetic-Resonance-Imaging <a name="magnetic-resonance-imaging"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-08-14

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	27
>- Last updated: 2026-09-22

### Medical-Physics <a name="medical-physics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	27
>- Last updated: 2026-09-22

### Simulation <a name="simulation"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	55 
>- Issues:	89
>- Watchers:	225
>- Last updated: 2026-10-19

- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	12
>- Last updated: 2026-05-03

### Super-Resolution <a name="super-resolution"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

### Brain-Connectivity <a name="brain-connectivity"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Fastmri-Challenge <a name="fastmri-challenge"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	50 
>- Issues:	6
>- Watchers:	320
>- Last updated: 2026-08-29

### Mri-Reconstruction <a name="mri-reconstruction"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	50 
>- Issues:	6
>- Watchers:	320
>- Last updated: 2026-08-29

### Medical-Image-Analysis <a name="medical-image-analysis"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Tensorflow <a name="tensorflow"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-08-14

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	7
>- Last updated: 2026-01-21

### Computer-Vision <a name="computer-vision"></a>
- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	130 
>- Issues:	17
>- Watchers:	872
>- Last updated: 2026-08-21

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	35 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Dicom-Converter <a name="dicom-converter"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	53 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-09-21

### Dicom-Images <a name="dicom-images"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	53 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-09-21

### Medical <a name="medical"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	53 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-09-21

### Medical-Images <a name="medical-images"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	53 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-09-21

### Png <a name="png"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	53 
>- Issues:	5
>- Watchers:	145
>- Last updated: 2026-09-21

### Afni-Brik-Head <a name="afni-brik-head"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Cifti-2 <a name="cifti-2"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Data-Formats <a name="data-formats"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Ecat <a name="ecat"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Gifti <a name="gifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Minc <a name="minc"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Streamlines <a name="streamlines"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Tck <a name="tck"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Trk <a name="trk"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	308 
>- Issues:	172
>- Watchers:	798
>- Last updated: 2026-10-26

### Mri-Phantoms <a name="mri-phantoms"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	15 
>- Issues:	22
>- Watchers:	35
>- Last updated: 2026-10-08

### Bart-Toolbox <a name="bart-toolbox"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	177 
>- Issues:	18
>- Watchers:	370
>- Last updated: 2026-07-27

### Compressed-Sensing <a name="compressed-sensing"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	177 
>- Issues:	18
>- Watchers:	370
>- Last updated: 2026-07-27

### Computational-Imaging <a name="computational-imaging"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	177 
>- Issues:	18
>- Watchers:	370
>- Last updated: 2026-07-27

### Iterative-Methods <a name="iterative-methods"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	177 
>- Issues:	18
>- Watchers:	370
>- Last updated: 2026-07-27

### Glioblastoma <a name="glioblastoma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	108
>- Last updated: 2026-10-20

### Glioma <a name="glioma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	108
>- Last updated: 2026-10-20

### 3D-Printing <a name="3d-printing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### 3D-Slicer <a name="3d-slicer"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### Computed-Tomography <a name="computed-tomography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### Image-Guided-Therapy <a name="image-guided-therapy"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### Kitware <a name="kitware"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### National-Institutes-Of-Health <a name="national-institutes-of-health"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### Nih <a name="nih"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### Qt <a name="qt"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### Tcia-Dac <a name="tcia-dac"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### Vtk <a name="vtk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	798 
>- Issues:	659
>- Watchers:	2681
>- Last updated: 2026-10-25

### 3D-Convolutional-Network <a name="3d-convolutional-network"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Brats2018 <a name="brats2018"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Brats2019 <a name="brats2019"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Densenet <a name="densenet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Iseg <a name="iseg"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Iseg-Challenge <a name="iseg-challenge"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Medical-Image-Segmentation <a name="medical-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Mrbrains18 <a name="mrbrains18"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Resnet <a name="resnet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Segmentation-Models <a name="segmentation-models"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Unet <a name="unet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### Unet-Image-Segmentation <a name="unet-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	22
>- Watchers:	1921
>- Last updated: 2026-10-15

### 3D-Slicer-Extension <a name="3d-slicer-extension"></a>
- [ukftractography](https://github.com/pnlbwh/ukftractography)
>- None

>- License: Other
>- Languages: `C`
>- Tags: 3d-slicer-extension
>- Forks:	31 
>- Issues:	10
>- Watchers:	30
>- Last updated: 2026-08-14

### Csharp <a name="csharp"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Image-Analysis <a name="image-analysis"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Java <a name="java"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Lua <a name="lua"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Ruby <a name="ruby"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Simpleitk <a name="simpleitk"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Swig <a name="swig"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Tcl <a name="tcl"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	236 
>- Issues:	99
>- Watchers:	1092
>- Last updated: 2026-10-23

### Reconstruction <a name="reconstruction"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	7
>- Watchers:	79
>- Last updated: 2026-10-23

### Retrospecitve <a name="retrospecitve"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	7
>- Watchers:	79
>- Last updated: 2026-10-23

### Slice-To-Volume <a name="slice-to-volume"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	9 
>- Issues:	7
>- Watchers:	79
>- Last updated: 2026-10-23

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

### Image-Segmentation <a name="image-segmentation"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	48 
>- Issues:	84
>- Watchers:	169
>- Last updated: 2026-10-06

### Structural-Mri <a name="structural-mri"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	48 
>- Issues:	84
>- Watchers:	169
>- Last updated: 2026-10-06

### Surface-Reconstruction <a name="surface-reconstruction"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	48 
>- Issues:	84
>- Watchers:	169
>- Last updated: 2026-10-06

### Pet-Mr <a name="pet-mr"></a>
- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	32 
>- Issues:	157
>- Watchers:	77
>- Last updated: 2026-10-15

### Freesurfer <a name="freesurfer"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	303 
>- Issues:	27
>- Watchers:	881
>- Last updated: 2026-10-26

### Lcn <a name="lcn"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	303 
>- Issues:	27
>- Watchers:	881
>- Last updated: 2026-10-26

### Denoising-Images <a name="denoising-images"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	64 
>- Issues:	80
>- Watchers:	195
>- Last updated: 2026-10-25

### Distortion-Correction <a name="distortion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	64 
>- Issues:	80
>- Watchers:	195
>- Last updated: 2026-10-25

### Motion-Correction <a name="motion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	64 
>- Issues:	80
>- Watchers:	195
>- Last updated: 2026-10-25

### Pipelines <a name="pipelines"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	64 
>- Issues:	80
>- Watchers:	195
>- Last updated: 2026-10-25

### Magnetic-Field-Solver <a name="magnetic-field-solver"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	27
>- Last updated: 2026-09-22

### Nmr <a name="nmr"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	27
>- Last updated: 2026-09-22

### Physics <a name="physics"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	11 
>- Issues:	4
>- Watchers:	27
>- Last updated: 2026-09-22

### 3D-Segmentation <a name="3d-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

### Frontend-App <a name="frontend-app"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

### Javascript <a name="javascript"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

### Mri-Segmentation <a name="mri-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

### Pyodide <a name="pyodide"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

### Tensorflowjs <a name="tensorflowjs"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

### Three-Js <a name="three-js"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	65 
>- Issues:	10
>- Watchers:	577
>- Last updated: 2026-10-26

### Cuda <a name="cuda"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	7
>- Watchers:	30
>- Last updated: 2026-06-16

### Gpu-Computing <a name="gpu-computing"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	7
>- Watchers:	30
>- Last updated: 2026-06-16

### Monte-Carlo-Simulation <a name="monte-carlo-simulation"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	7
>- Watchers:	30
>- Last updated: 2026-06-16

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

### Mri-Sequences <a name="mri-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	99 
>- Issues:	38
>- Watchers:	221
>- Last updated: 2026-10-20

### Pulse-Sequences <a name="pulse-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	99 
>- Issues:	38
>- Watchers:	221
>- Last updated: 2026-10-20

### Pulseq <a name="pulseq"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	99 
>- Issues:	38
>- Watchers:	221
>- Last updated: 2026-10-20

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

### Cortical-Thickness <a name="cortical-thickness"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	34
>- Last updated: 2026-08-06

### Morphometry <a name="morphometry"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	7 
>- Issues:	4
>- Watchers:	34
>- Last updated: 2026-08-06

### Analysis <a name="analysis"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	15 
>- Issues:	5
>- Watchers:	41
>- Last updated: 2026-09-30

### Cancer-Imaging-Research <a name="cancer-imaging-research"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	15 
>- Issues:	5
>- Watchers:	41
>- Last updated: 2026-09-30

### Dce-Mri <a name="dce-mri"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	15 
>- Issues:	5
>- Watchers:	41
>- Last updated: 2026-09-30

### Mat-Files <a name="mat-files"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	15 
>- Issues:	5
>- Watchers:	41
>- Last updated: 2026-09-30

### Bids-Apps <a name="bids-apps"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

### Nipype <a name="nipype"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

### Workflow <a name="workflow"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	33
>- Last updated: 2026-07-24

### Complex-Networks <a name="complex-networks"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Connectome <a name="connectome"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Connectomics <a name="connectomics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Graph-Theory <a name="graph-theory"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Network-Analysis <a name="network-analysis"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Statistics <a name="statistics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	52 
>- Issues:	11
>- Watchers:	200
>- Last updated: 2026-10-12

### Cardiac <a name="cardiac"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	55 
>- Issues:	89
>- Watchers:	225
>- Last updated: 2026-10-19

### Diffusion <a name="diffusion"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	55 
>- Issues:	89
>- Watchers:	225
>- Last updated: 2026-10-19

### Gpu-Acceleration <a name="gpu-acceleration"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a GPU-accelerated, Pulseq-compatible framework for simulating Magnetic Resonance Imaging (MRI) acquisitions. The primary focus of this package is to simulate the wide range of scenarios encountered during pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	55 
>- Issues:	89
>- Watchers:	225
>- Last updated: 2026-10-19

### Neural-Networks <a name="neural-networks"></a>
- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	341 
>- Issues:	23
>- Watchers:	1068
>- Last updated: 2026-10-22

### Fastmri <a name="fastmri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

### Fastmri-Dataset <a name="fastmri-dataset"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	422 
>- Issues:	18
>- Watchers:	1537
>- Last updated: 2026-10-25

### Brain-Mri <a name="brain-mri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

### Decoding <a name="decoding"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

### Mvpa <a name="mvpa"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	691 
>- Issues:	278
>- Watchers:	1452
>- Last updated: 2026-10-25

### Augmentation <a name="augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

### Data-Augmentation <a name="data-augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, python, pytorch
>- Forks:	288 
>- Issues:	60
>- Watchers:	2450
>- Last updated: 2026-10-23

### Fcm <a name="fcm"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

### Harmonization <a name="harmonization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

### Intensity-Normalization <a name="intensity-normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

### Normalization <a name="normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

### Standardization <a name="standardization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

### Whitestripe <a name="whitestripe"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

### Zscore <a name="zscore"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	57 
>- Issues:	3
>- Watchers:	340
>- Last updated: 2026-08-17

### Healthcare-Imaging <a name="healthcare-imaging"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1685 
>- Issues:	522
>- Watchers:	8804
>- Last updated: 2026-10-26

### Monai <a name="monai"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1685 
>- Issues:	522
>- Watchers:	8804
>- Last updated: 2026-10-26

### Python3 <a name="python3"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1685 
>- Issues:	522
>- Watchers:	8804
>- Last updated: 2026-10-26

### 3D-Reconstruction <a name="3d-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### 3D-Visualization <a name="3d-visualization"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### Implicit-Neural-Representation <a name="implicit-neural-representation"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### Nerf <a name="nerf"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### Neural-Network <a name="neural-network"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### Neural-Rendering <a name="neural-rendering"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### Transformers <a name="transformers"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	26 
>- Issues:	7
>- Watchers:	117
>- Last updated: 2026-10-01

### Ml <a name="ml"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	11
>- Watchers:	49
>- Last updated: 2026-08-14

### Imaging <a name="imaging"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

### Quality-Metrics <a name="quality-metrics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	41 
>- Issues:	2
>- Watchers:	126
>- Last updated: 2026-10-22

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
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### 3D-Models <a name="3d-models"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### 3D-Object-Detection <a name="3d-object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Deep-Neural-Networks <a name="deep-neural-networks"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Detection <a name="detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Mask-Rcnn <a name="mask-rcnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Object-Detection <a name="object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Pytorch-Cnn <a name="pytorch-cnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Pytorch-Deeplearning <a name="pytorch-deeplearning"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Pytorch-Implementation <a name="pytorch-implementation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Retina-Net <a name="retina-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Retina-Unet <a name="retina-unet"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Semantic-Segmentation <a name="semantic-segmentation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### U-Net <a name="u-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	293 
>- Issues:	49
>- Watchers:	1362
>- Last updated: 2026-10-25

### Neuroimage <a name="neuroimage"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	123 
>- Issues:	431
>- Watchers:	274
>- Last updated: 2026-10-19

### Spinalcord <a name="spinalcord"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	123 
>- Issues:	431
>- Watchers:	274
>- Last updated: 2026-10-19

### Quality-Reporter <a name="quality-reporter"></a>
- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	142 
>- Issues:	101
>- Watchers:	381
>- Last updated: 2026-10-14

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

### Inverse-Problems <a name="inverse-problems"></a>
- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	50 
>- Issues:	6
>- Watchers:	320
>- Last updated: 2026-08-29

### Denoising-Algorithm <a name="denoising-algorithm"></a>
- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	25
>- Last updated: 2026-09-09

### Big-Data <a name="big-data"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

### Brainweb <a name="brainweb"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

### Data-Science <a name="data-science"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

### Dataflow <a name="dataflow"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

### Dataflow-Programming <a name="dataflow-programming"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14

### Workflow-Engine <a name="workflow-engine"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	541 
>- Issues:	426
>- Watchers:	835
>- Last updated: 2026-10-14



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
