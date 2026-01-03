---
title: 'Spyctra and pyTNMR: Python Tools For Magnetic Resonance Analysis and Control'
tags:
  - python
  - spectroscopy
  - NMR
  - NQR
  - FFC
  - DNP
  - EPR
  - magnetic resonance
authors:
  - name: Michael W. Malone
    corresponding: true
    orcid: 0000-0002-9721-1538
    affiliation: 1
  - name: Adam R Altenhof
    orcid: 0000-0002-8095-6373
    affiliation: 1
  - name: Harris E. Mason
    orcid: 0000-0002-1840-0550
    affiliation: 1
  - name: Martin Anderson
    affiliation: 2
affiliations:
 - name: Los Alamos National Laboratory, Los Alamos, NM, US
   index: 1
 - name: Anderson Intuition LLC, Berkeley, CA, US
   index: 2
date: 3 January 2026
bibliography: paper.bib
---

# Summary

The field of magnetic resonance includes nuclear magnetic resonance (NMR), magnetic resonance imaging (MRI), electron paramagnetic resonance (EPR), nuclear quadrupole resonance (NQR), and specialist techniques like fast field cycling (FFC) NMR, dynamic nuclear polarization (DNP), solid state NMR (SSNMR), etc. Despite the broad community and applications, the structure of the raw data is, generally, sets of uniformly sampled signals, and is processed with similar, if not identical, procedures. This means a single data processing approach can be useful across the breadth of magnetic resonance researchers. In particular, efficient numerical tools that can provide rapid yet deep analyses of data as it is taken is necessary across all aspects of magnetic resonance to avoid wasting limited, or costly, experimental time.

# Statement of need

`Spyctra` provides an entirely open source scripting approach to automate the processing of magnetic resonance data. It is an especially powerful tool when utilized concurrently during an experimental collection (e.g. when used with `pyTNMR` to control Tecmag Inc.'s spectrometers). The result is that the experimentalist can ask, and answer, questions about their data as it is collected that might otherwise be unavailable, or difficult to automatically track. This provides rapid feedback to the user and helps avoid unnecessary or failed experiments, which prevents wasting time or other resources.

The benefits of a scripting approach, compared to a graphical user interface (GUI) approach, to data processing are numerous and significant. 1) Scripts mean data processing is reproducible, documented, and shareable. Mistakes are inevitable, but being able to quickly rewrite and redo an analysis means experiments quickly converge to the desired outcome. 2) Scripts scale. As more is learned about an experiment during the collect, or over years of repeating similar experiments, scripts naturally grow in complexity as more parameters are tracked or better processing algorithms are developed. Scripting allows for the reprocessing of data that would be extremely laborious, and error prone, to complete with a point and click interface. 3) Scripts encourage investigation. Because `spyctra` is written in python, methods can easily be altered, and new methods tried, with no limits in how the data can be processed, combined with other measurements from other inputs, shared with other programs, moved across file systems, or used to control other instruments.

One obvious drawback is that GUI based approaches help visually confirm processing is progressing as predicted. However `spyctra` is built with several tool to rapidly help visualize data, including the abilty to directly compare data before and after each step. In addition, `spcytra` includes a fitting library, fitlib, that can visualize fits, including global fits of complex data, to help converge on the correct fit parameters.

`pyTNMR` provides a prototypical interface between python and a magnetic resonance spectrometer. It handles Tecmag Inc.'s ".tnt" file creation, manipulation, and automation of experimental collections. Because `pyTNMR` naturally interacts with `spyctra`, experiments can be made with active feedback based on any result from a previous measurement. For instance, this means the resonance frequency can be measured in experiment #1 that is then used as the observation frequency in experiment #2 to minimize off-resonant effects.

In regular development for over ten years, both `spyctra` and `pyTNMR` have been used for publications across many magnetic resonance experiments including low and Earth's field NMR (@malone2016vivo, @Kaseman_Jouster); NQR detection and FFC analysis of fentanyl compounds (@Malone_HCl, @Malone_citrate); and DNP (@ALTENHOF_DNP). This shows its utility to the broad community of magnetic resonance researchers.

# Acknowledgements

Research presented in this report was partially supported by the Laboratory Directed Research and Development program of Los Alamos National Laboratory under project number 20260021DR. Los Alamos National Laboratory is operated by Triad National Security, LLC, for the National Nuclear Security Administration of US Department of Energy (contract no. 89233218CNA000001).

# References
