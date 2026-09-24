# First Launch and Sample Graphs

After downloading the Windows prebuilt package, the fastest way to understand F8Studio is to open a graph guide and build the graph step by step.

## Step 1: Download a Prebuilt Package

Get the latest Windows package from:

[GitHub Releases](https://github.com/feel8-fun/f8studio/releases){ .md-button .md-button--primary }

## Step 2: Open Studio

Launch `f8studio.exe` on Windows or `./f8studio` on Linux from the extracted release folder. The launcher starts the local server and opens Web Studio in your default browser.

## Step 3: Pick a Graph Guide

The graph guide library contains practical workflows you can recreate in Studio:

1. [CVKit Template Tracking](../scenarios/cvkit-template-tracking.md)
2. [Audio Driven TCode](../scenarios/audio-driven-tcode.md)
3. [Functional TCode Generation](../scenarios/functional-tcode-generation.md)

Each guide explains the node chain, key parameters, validation checks, and any copy-paste script blocks needed for that workflow.

## Step 4: Build the Graph in Studio

Add the nodes listed in the guide, wire them in the documented order, then review the required state fields before starting services.

## Step 5: Run It Carefully

1. Start infrastructure/runtime host services first
2. Start producers such as media or capture services
3. Start downstream processing/output nodes
4. Watch status, logs, and visualization nodes

## What To Read Next

- [Studio Quickstart](../getting-started/studio.md)
- [For Graph Authors](../graph-authors/index.md)
