# To-Do List

- [ ] Add a legend to the 3D plots
- [ ] Add a recording feature to the 3D plots
- [ ] Move FTE and EKF algorithms into their own modules
- [ ] Add a GUI so that users can run optimizations and perform calibrations without needing to modify any jupyter notebooks and/or code
- [ ] Add features to [lib.extract.VideoLabelSession](https://github.com/African-Robotics-Unit/AcinoSet/blob/69eed4795cbc163b0f8979f5f89b6d1a381765bc/src/lib/extract.py#L58) to the point where Argus Clicker is no longer required
- [ ] Write algorithm to quantify the accuracy/goodness?:neckbeard: of a reconstruction
- [ ] Write algorithm that automatically adjusts reconstruction params (start_frame, end_frame, dlc_thresh & perhaps the link lengths in [lib.misc.get_3d_marker_coords](https://github.com/African-Robotics-Unit/AcinoSet/blob/69eed4795cbc163b0f8979f5f89b6d1a381765bc/src/lib/misc.py#L34)) to obtain the most accurate reconstruction for a given run/flick.
- [ ] Make [lib.calib.adjust_extrinsics_manual_points](https://github.com/African-Robotics-Unit/AcinoSet/blob/69eed4795cbc163b0f8979f5f89b6d1a381765bc/src/lib/calib.py#L215) more robust by automatically adjusting the redescending loss params