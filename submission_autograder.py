#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))
