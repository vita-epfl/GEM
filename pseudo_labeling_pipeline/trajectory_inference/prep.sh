#!/bin/bash

echo "Installing geocalib..."
cd GeoCalib
python3 -m pip install -e .
cd ..
echo "Done."

pip3 install -e .
# cd droid_trajectory/droid_slam && python3 setup.py install
# cd ../..

"$@"