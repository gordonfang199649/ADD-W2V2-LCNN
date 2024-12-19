

python3 .\main_train_in_the_wild.py --num_epochs 10 --out_fold "./models/in_the_wild/" 

python3 .\generate_score.py
python3 .\generate_score.py --task "in_the_wild"
python3 .\generate_score.py --model_folder "./models/in_the_wild" --task "in_the_wild"
python3 .\generate_score.py --model_folder "./models/in_the_wild"

python3 .\main_train_in_the_wild_asvspoof.py --num_epochs 10 --out_fold "./models/in_the_wild_asvspoof2019/" 
python3 .\generate_score.py --model_folder "./models/in_the_wild_asvspoof2019"
python3 .\generate_score.py --model_folder "./models/in_the_wild_asvspoof2019" --task "in_the_wild"



python3 .\generate_score.py --task "accentdb"

python3 .\generate_score.py --model_folder "./models/in_the_wild" --task "accentdb"

python3 .\main_train_new.py --num_epochs 10 --out_fold "./models/DFADD/"  --name "DFADD"
python3 .\main_train_new.py --num_epochs 10 --out_fold "./models/ASVspoof2021_DF/"  --name "ASVspoof2021_DF"


python3 .\main_train_new.py --num_epochs 10 --out_fold "./models/DFADD/"  --name "DFADD" --upsample_num 35785 --downsample_num 0
python3 .\main_train_new.py --num_epochs 10 --out_fold "./models/ASVspoof2021_DF/"  --name "ASVspoof2021_DF" --upsample_num 104014 --downsample_num 235146
python3 .\main_train_new.py --num_epochs 10 --out_fold "./models/CodecFake/"  --name "CodecFake" --upsample_num 104014 --downsample_num 265916 
python3 .\main_train_new_DCA.py --num_epochs 10 --out_fold "./models/DCA/"  --name "DCA"

python3 .\generate_score_new.py --model_folder "./models/DFADD" --task "DFADD"
python3 .\generate_score_new.py --model_folder "./models/ASVspoof2021_DF" --task "ASVspoof2021_DF"
python3 .\generate_score_new.py --model_folder "./models/CodecFake" --task "CodecFake"
python3 .\generate_score_new.py --model_folder "./models/DCA" --task "DFADD"
python3 .\generate_score_new.py --model_folder "./models/DCA" --task "ASVspoof2021_DF"
python3 .\generate_score_new.py --model_folder "./models/DCA" --task "CodecFake"
python3 .\generate_score_new.py --model_folder "./models/DCA" --task "in_the_wild"

python3 .\generate_score_new.py --model_folder "./models/in_the_wild" --task "in_the_wild"
