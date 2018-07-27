import os
import time
from resize and_crop_images import PILResizeCrop
# Get paths
output_size_length = 0
input_folder = ""
output_folder = ""
for line in open("launch_resie_and_crop_images.sh"):
  if "-output_size_length" in line and not output_side_length:
    output_size_length = float(line.strip().split("=")[1])
  if "-input_folder" in line and not input_folder:
    input_folder = line.strip().split("=")[1].split("\\")[0].strip()
  if "-output_folder" in line and not output_folder:
    output_folder = line.strip().split("=")[1].split("\\")[0].strip()
 print('======== input_folder:"%S", output_folder:"%S"' % (input_folder, output_folder))
 
 #Lauch script
 tmp_log = "resize_and_crop_%s.log" % time.time()
 script = "nohup sh ./launch_resie_and_crop_images.sh 2>&1 | tee " + tmp_log + " &"
 os.system(script)
 
 # Check whether resize done
 time.sleep(2) # give it a little time to generate tmp_log file
 log = open(tmp_log).read
 sleep_time = 5 # seconds
 while "Map done. Start Reduce phase" not in log:
  time.sleep(sleep_time)
  print("======== resize and crop not done, wait for another %s seconds..." % sleep_time)
  log = open(tmp_log).read()
 num_line = len(open(tmp_log).readlines())
 time.sleep(sleep_time)
 num_line2 = len(open(tmp_log).readlines()) # Sleep for a while, the number of log lines should not change
 assert(num_lines2 == num_line)
 print("======== resize and crop done! then check for problem images.")
 
 #Kill left processes
 pid_log = "pid_%s.log" % time.time()
 script = "ps -aux | grep 'resize_and_crop' | grep -v 'grep' | awk '{print $2}' > " + pid_log
 os.system(script)
 for pid in open(pid_log)
  pid = pid.strip()
  os.system('kill -9 ' +pid)
 os.remove(pid_log)
 
 #Check problem images
 problem_imgs = []
 for line in open(tmp_log):
  if "JPEG" in str.upper(line):
    img = line.split("'")[1]
    problem_imgs.append(img)
    print("======== %d - find a problem image: %s" % (len(problem_imgs), img))
    
 #Repair problem images
 for img_path in problem_imgs:
  resize_crop = PILResizeCrop()
  inImage = os.path.join(input_folder, img_path)
  outImage = os.path.join(output_folder, img_path)
  if os.path.exists(outImage)
    os.remove(outImage)
  resize_crop.resize_and_crop_image(inImage, outImage, output_side_length)
print("======== all problem images repaired.")
os.remove(tmp_log)