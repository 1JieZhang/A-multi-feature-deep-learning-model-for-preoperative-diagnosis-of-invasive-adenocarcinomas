1.输入医生收集的病人信息xlsx表格地址，模板在文件夹内部，second_A_group和second_B_group

2.输入为CT图像的地址，保证一个子文件夹内含有一个病人的data和mask信息，见Input data stores styles

3.需要调整patient_number = patient_dir.split('\\')[-1][:7]中的范围，保证输出为医生收集的病人的病号

4.输出样板和数据存放样式在文件夹内部