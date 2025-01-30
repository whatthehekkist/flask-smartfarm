#!/bin/bash
gunicorn app:app
#gunicorn app:app --bind 0.0.0.0:10000
