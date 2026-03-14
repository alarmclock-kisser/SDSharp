# Copilot Instructions

## General Guidelines
- In async methods, prefer awaiting `StaticLogger.LogAsync` instead of using `StaticLogger.Log` directly.

## Project-Specific Rules
- Do not use placeholder or generic default landscape rendering for Stable Diffusion; implement real Stable Diffusion inference end-to-end instead of mock image generation.