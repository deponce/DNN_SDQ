export num_examples=10000
# python3 get_DCTgrad.py -model Alexnet -Batch_size 128 -Nexample ${num_examples}
python3 get_DCTgrad.py -model VGG11 -Batch_size 128 -Nexample ${num_examples}
python3 get_DCTgrad.py -model Resnet18 -Batch_size 128 -Nexample ${num_examples}
python3 get_DCTgrad.py -model Squeezenet -Batch_size 128 -Nexample ${num_examples}