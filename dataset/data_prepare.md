## Prepare data for Kinetics400

- Download raw videos from [official website](https://deepmind.com/research/open-source/kinetics). The data structure should be like below:

  ```
  kinetics/
  ├── raw_video/
  │   ├── abseiling/
  │   │   ├──0347ZoDXyP0_000095_000105.mp4
  │   │   ├──035LtPeUFTE_000085_000095.mp4
  │   │   ├──......
  │   ├── air_druming/
  │   │   ├──03V2idM7_KY_000003_000013.mp4
  │   │   ├──0aidJ-1R7Ds_000003_000013.mp4
  │   │   ├──......
  │   ├──......
  ```

- Extract raw frames:

  ```bash
  python3 dataset/process_data/create_lmdb.py --root-path data/kinetics/raw_video --dst-path data/kinetics/rgb_lmdb --num-workers -1 --data-type rgb --dataset kinetics
  ```

- Extract optical flows:

  ```bash
  python3 dataset/process_data/create_lmdb.py --root-path data/kinetics/rgb_lmdb --dst-path data/kinetics/flow_lmdb --num-workers -1 --data-type flow --dataset kinetics
  ```

- Extract motion magnitude for each frame:

  ```bash
  python3 dataset/process_data/create_lmdb.py --root-path data/kinetics/flow_lmdb --dst-path data/kinetics/flow_mag_lmdb --num-workers -1 --data-type mag --dataset kinetics
  ```

- Extract motion magnitude for each clip

  ```shell
  python3 dataset/process_data/create_lmdb.py --root-path data/kinetics/flow_lmdb --dst-path data/kinetics/ --num-workers -1 --data-type clip-mag --dataset kinetics
  ```

- The final data structure should like this:

  ```
  kinetics/
  ├── raw_video/
  ├── ├── ......
  ├── rgb_lmdb/
  ├── ├── ......
  ├── flow_lmdb/
  ├── ├── ......
  ├── flow_mag_lmdb/
  ├── ├── ......
  ├── video_clip_mag_kinetics.pickle
  ```
  
## Prepare data for UCF101

- Download raw videos from the [official website](https://www.crcv.ucf.edu/data/UCF101.php). The data structure should be like below:

  ```
  ucf101/
  ├── raw_video/
  │   ├── ApplyEyeMakeup/
  │   │   ├──v_ApplyEyeMakeup_g01_c01.avi
  │   │   ├──v_ApplyEyeMakeup_g01_c02.avi
  │   │   ├──......
  │   ├── ApplyLipstick/
  │   │   ├──v_ApplyLipstick_g01_c01.avi
  │   │   ├──v_ApplyLipstick_g01_c02.avi
  │   │   ├──......
  │   ├──......
  ```


- Then you can prepare data for UCF101 with same procedure above and change the `--dataset` to  `ucf101`.  The final data structure of UCF101 should like this:

  ```
  ucf101/
  ├── raw_video/
  ├── ├── ......
  ├── rgb_lmdb/
  ├── ├── ......
  ├── flow_lmdb/
  ├── ├── ......
  ├── flow_mag_lmdb/
  ├── ├── ......
  ├── video_clip_mag_ucf101.pickle
  ```
