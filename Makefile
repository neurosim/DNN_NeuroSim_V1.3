make:
	cd Inference_pytorch/NeuroSIM && make
	cd Inference_pytorch && ./NeuroSIM/main ./NeuroSIM/NetWork_ResNet18.csv 8 8 ./layer_record_ResNet18/weightConv_0_.csv ./layer_record_ResNet18/inputConv_0_.csv ./layer_record_ResNet18/weightConv3x3_1_.csv ./layer_record_ResNet18/inputConv3x3_1_.csv ./layer_record_ResNet18/weightConv3x3_2_.csv ./layer_record_ResNet18/inputConv3x3_2_.csv ./layer_record_ResNet18/weightConv3x3_3_.csv ./layer_record_ResNet18/inputConv3x3_3_.csv ./layer_record_ResNet18/weightConv3x3_4_.csv ./layer_record_ResNet18/inputConv3x3_4_.csv ./layer_record_ResNet18/weightConv3x3_6_.csv ./layer_record_ResNet18/inputConv3x3_6_.csv ./layer_record_ResNet18/weightConv3x3_7_.csv ./layer_record_ResNet18/inputConv3x3_7_.csv ./layer_record_ResNet18/weightConv1x1_5_.csv ./layer_record_ResNet18/inputConv1x1_5_.csv ./layer_record_ResNet18/weightConv3x3_8_.csv ./layer_record_ResNet18/inputConv3x3_8_.csv ./layer_record_ResNet18/weightConv3x3_9_.csv ./layer_record_ResNet18/inputConv3x3_9_.csv ./layer_record_ResNet18/weightConv3x3_11_.csv ./layer_record_ResNet18/inputConv3x3_11_.csv ./layer_record_ResNet18/weightConv3x3_12_.csv ./layer_record_ResNet18/inputConv3x3_12_.csv ./layer_record_ResNet18/weightConv1x1_10_.csv ./layer_record_ResNet18/inputConv1x1_10_.csv ./layer_record_ResNet18/weightConv3x3_13_.csv ./layer_record_ResNet18/inputConv3x3_13_.csv ./layer_record_ResNet18/weightConv3x3_14_.csv ./layer_record_ResNet18/inputConv3x3_14_.csv ./layer_record_ResNet18/weightConv3x3_16_.csv ./layer_record_ResNet18/inputConv3x3_16_.csv ./layer_record_ResNet18/weightConv3x3_17_.csv ./layer_record_ResNet18/inputConv3x3_17_.csv ./layer_record_ResNet18/weightConv1x1_15_.csv ./layer_record_ResNet18/inputConv1x1_15_.csv ./layer_record_ResNet18/weightConv3x3_18_.csv ./layer_record_ResNet18/inputConv3x3_18_.csv ./layer_record_ResNet18/weightConv3x3_19_.csv ./layer_record_ResNet18/inputConv3x3_19_.csv ./layer_record_ResNet18/weightFC_20_.csv ./layer_record_ResNet18/inputFC_20_.csv | tee neurosim_out.txt
	make get_output

get_output:
	echo "ResNet18 Layer Shapes: "
	cat Inference_pytorch/NeuroSIM/NetWork_ResNet18.csv
	echo ""
	echo "ResNet18 Average Inputs/Weights: "
	python3 avg_input_weight.py

	echo "Per-Layer Energy: "
	python3 get_energy.py

	cat Inference_pytorch/neurosim_out.txt | grep "Total Run-time of NeuroSim:"