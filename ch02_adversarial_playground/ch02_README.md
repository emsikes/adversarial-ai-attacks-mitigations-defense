# Chapter 02 — Building Our Adversarial Playground

## Overview
Environment hardening and security tooling setup across the adversarial lab.
Establishes the secure baseline all subsequent attack chapters operate from.

## Key Tools
- ModelScan — serialization vulnerability detection
- Bandit — Python static code security analysis
- Safety — dependency vulnerability scanning
- detect-secrets — secret scanning
- Jupyter security hardening

## Files
- security_scan.py   — ModelScan, Bandit, and Safety scan wrappers
- secrets_scan.py    — detect-secrets workflow
- verify_env.py      — full environment verification script