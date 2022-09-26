#!/usr/bin/env bash

openssl genrsa -out key.pem
openssl req -new -key key.pem -out csr.pem -config csr.conf
openssl x509 -req -days 9999 -in csr.pem -signkey key.pem -out cert.pem -extensions req_ext -extfile csr.conf
rm csr.pem
