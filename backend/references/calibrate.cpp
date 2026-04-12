// Sportlight keypoint homography estimation

#include <fstream>
#include <iostream>
#include <map>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <nlohmann/json.hpp>
#include <filesystem>

using namespace std;

// Semantic mapping from TVCalib/SoccerNet keypoint class names -> 2D pitch coordinates (meters).
// IMPORTANT CONTRACT:
// - `track_teams.py` expects the inverse homography (image->pitch) to yield meters.
// - It then applies: radar_x = (x_m + 52.5) * 10, radar_y = (y_m + 34.0) * 10.
// So the values below must be physical meters with soccer center (0,0).
map<string, cv::Point2f> pitch_template = {
    // Center
    {"CenterMark", cv::Point2f(0.0f, 0.0f)},

    // Corners
    {"TL_PITCH_CORNER", cv::Point2f(-52.5f, -34.0f)},
    {"TR_PITCH_CORNER", cv::Point2f(52.5f, -34.0f)},
    {"BL_PITCH_CORNER", cv::Point2f(-52.5f, 34.0f)},
    {"BR_PITCH_CORNER", cv::Point2f(52.5f, 34.0f)},

    // Penalty area
    {"L_PENALTY_AREA_TR_CORNER", cv::Point2f(-36.0f, -20.16f)},
    {"L_PENALTY_AREA_BR_CORNER", cv::Point2f(-36.0f, 20.16f)},
    {"R_PENALTY_AREA_TL_CORNER", cv::Point2f(36.0f, -20.16f)},
    {"R_PENALTY_AREA_BL_CORNER", cv::Point2f(36.0f, 20.16f)},

    // Goal area
    {"L_GOAL_AREA_TR_CORNER", cv::Point2f(-47.0f, -9.16f)},
    {"L_GOAL_AREA_BR_CORNER", cv::Point2f(-47.0f, 9.16f)},
    {"R_GOAL_AREA_TL_CORNER", cv::Point2f(47.0f, -9.16f)},
    {"R_GOAL_AREA_BL_CORNER", cv::Point2f(47.0f, 9.16f)},

    // Goal posts
    {"L_GOAL_TL_POST", cv::Point2f(-52.5f, 3.66f)},
    {"L_GOAL_TR_POST", cv::Point2f(-52.5f, -3.66f)},
    {"R_GOAL_TL_POST", cv::Point2f(52.5f, -3.66f)},
    {"R_GOAL_TR_POST", cv::Point2f(52.5f, 3.66f)},

    // Mid touch intersections
    {"T_TOUCH_AND_HALFWAY_LINES_INTERSECTION", cv::Point2f(0.0f, -34.0f)},
    {"B_TOUCH_AND_HALFWAY_LINES_INTERSECTION", cv::Point2f(0.0f, 34.0f)},
};

static double computeConditionNumber(const cv::Mat& H64) {
  cv::SVD svd(H64);
  // Singular values are sorted descending; use sigma_max / sigma_min.
  if (svd.w.empty()) {
    return 0.0;
  }
  const double sigma_max = svd.w.at<double>(0);
  const double sigma_min = svd.w.at<double>(svd.w.rows - 1);
  if (std::abs(sigma_min) < 1e-12) {
    return 0.0;
  }
  return sigma_max / sigma_min;
}

bool isMatrixValid(const cv::Mat& H,
                    const vector<cv::Point2f>& src,
                    const vector<cv::Point2f>& dst,
                    double* out_avgErr,
                    double* out_cond) {
  if (H.empty()) {
    return false;
  }

  if (src.size() < 4 || dst.size() < 4 || src.size() != dst.size()) {
    return false;
  }

  cv::Mat H64;
  if (H.type() == CV_64F) {
    H64 = H;
  } else {
    H.convertTo(H64, CV_64F);
  }

  if (out_cond) {
    *out_cond = computeConditionNumber(H64);
  }

  // Reprojection check using the same src->dst correspondences we used to estimate H.
  vector<cv::Point2f> projected;
  cv::perspectiveTransform(src, projected, H64);
  if (projected.size() != dst.size()) {
    return false;
  }

  double sumErr = 0.0;
  for (size_t i = 0; i < projected.size(); i++) {
    sumErr += cv::norm(projected[i] - dst[i]);
  }
  const double avgErr = sumErr / static_cast<double>(projected.size());
  if (out_avgErr) {
    *out_avgErr = avgErr;
  }
  // Relaxed for intersection-point noise (and because dst is in a different scale).
  if (avgErr > 25.0) {
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  const string jsonPath = (argc > 1) ? string(argv[1]) : "calibration_data.json";

  ifstream f(jsonPath);
  if (!f.is_open()) {
    cerr << "Failed to open: " << jsonPath << endl;
    return 1;
  }

  nlohmann::json j;
  f >> j;

  if (!j.contains("frames") || !j["frames"].is_array()) {
    cerr << "Invalid JSON schema: expected {\"frames\": [...]} in " << jsonPath << endl;
    return 1;
  }

  const auto frames = j["frames"];
  if (frames.size() < 2) {
    cerr << "Not enough frames to estimate homographies." << endl;
    return 1;
  }

  cv::Mat last_good_matrix;
  bool has_last_good_matrix = false;

  cv::Mat smoothed_matrix;

  // Output JSON for the Python tracker (`backend/scripts/track_teams.py`).
  // The tracker expects pitch->image homographies in `homographies[]` and then inverts them.
  nlohmann::json output_json;
  output_json["homographies"] = nlohmann::json::array();

  // Video ingestion (to bridge AI blind spots with optical flow).
  cv::VideoCapture cap("backend/references/match_video.mp4");
  if (!cap.isOpened()) {
    cerr << "Failed to open video: backend/references/match_video.mp4" << endl;
    return 1;
  }

  cv::Mat current_frame;
  cv::Mat prev_gray;
  cv::Mat current_gray;
  vector<cv::Point2f> prev_pts;

  // Print a compact sample for the first frames only (keeps logs readable).
  size_t sample_logged = 0;
  const size_t sample_limit = 20;

  for (size_t i = 0; i + 1 < frames.size(); i++) {
    const auto& f0 = frames[i];

    // We estimate image->pitch homographies using semantic ID matching:
    // src comes from JSON keypoints (semantic dict), dst comes from pitch_template (fixed).
    if (!f0.contains("keypoints")) {
      continue;
    }

    vector<cv::Point2f> src;
    vector<cv::Point2f> dst;

    for (auto it = f0["keypoints"].begin(); it != f0["keypoints"].end(); ++it) {
      const string key = it.key();
      if (!pitch_template.count(key)) {
        continue;
      }

      const auto& kp = it.value();
      if (!kp.is_array() || kp.size() < 2) {
        continue;
      }

      const float x = kp[0].get<float>();
      const float y = kp[1].get<float>();

      // Skip non-finite points (we no longer use [0,0] padding in the exporter).
      if (!std::isfinite(x) || !std::isfinite(y)) {
        continue;
      }

      src.emplace_back(x, y);
      dst.emplace_back(pitch_template[key]);
    }

    cv::Mat H_candidate;
    bool candidate_valid = false;
    double avgErr = 0.0;
    double condNum = 0.0;

    bool used_fresh_matrix = false;

    // Load the current video frame (frame_idx from JSON).
    const long long f0_frame_idx = f0.contains("frame_idx") ? f0["frame_idx"].get<long long>() : -1LL;
    if (f0_frame_idx >= 0) {
      cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(f0_frame_idx));
      cap >> current_frame;
      if (current_frame.empty()) {
        cerr << "Failed to read frame_idx=" << f0_frame_idx << " from video." << endl;
        return 1;
      }
      cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
    } else {
      // If we don't have frame_idx, we can't optical-flow bridge this iteration.
      candidate_valid = false;
    }

    if (src.size() >= 4 && dst.size() >= 4 && src.size() == dst.size()) {
      try {
        cv::Mat inliersMask;
        // Use RANSAC mask so reprojection validation isn't poisoned by outliers.
        H_candidate = cv::findHomography(src, dst, cv::RANSAC, 3.0, inliersMask);

        vector<cv::Point2f> src_inliers;
        vector<cv::Point2f> dst_inliers;
        if (!inliersMask.empty()) {
          src_inliers.reserve(src.size());
          dst_inliers.reserve(dst.size());
          for (size_t k = 0; k < src.size(); k++) {
            // Mask is Nx1 CV_8U with 0/1 inliers.
            const uchar m = inliersMask.at<uchar>(static_cast<int>(k), 0);
            if (m) {
              src_inliers.push_back(src[k]);
              dst_inliers.push_back(dst[k]);
            }
          }
        } else {
          src_inliers = src;
          dst_inliers = dst;
        }

        if (src_inliers.size() >= 4) {
          candidate_valid = isMatrixValid(H_candidate, src_inliers, dst_inliers, &avgErr, &condNum);
        } else {
          candidate_valid = false;
        }
      } catch (const cv::Exception&) {
        candidate_valid = false;
      } catch (const std::exception&) {
        candidate_valid = false;
      } catch (...) {
        candidate_valid = false;
      }
    } else {
      candidate_valid = false;
    }

    // Track how matrix was produced for logging.
    string used_mode = "smoothed_fallback";

    if (candidate_valid) {
      // Anti-teleportation gate:
      // Reject AI matrices that imply an impossible camera jump within a single step.
      // We compare the projected pitch coordinates of the image center using the
      // current smoothed homography (history) vs the new candidate H.
      if (!smoothed_matrix.empty()) {
        // Homography coordinate system is the same as our keypoints: 1280x720.
        const cv::Point2f img_center_px(640.0f, 360.0f);
        std::vector<cv::Point2f> center_src{img_center_px};

        std::vector<cv::Point2f> smoothed_pitch;
        std::vector<cv::Point2f> candidate_pitch;
        cv::perspectiveTransform(center_src, smoothed_pitch, smoothed_matrix);
        cv::perspectiveTransform(center_src, candidate_pitch, H_candidate);

        if (!smoothed_pitch.empty() && !candidate_pitch.empty()) {
          const double dx = static_cast<double>(candidate_pitch[0].x - smoothed_pitch[0].x);
          const double dy = static_cast<double>(candidate_pitch[0].y - smoothed_pitch[0].y);
          const double jump_dist_m = std::sqrt(dx * dx + dy * dy);
          if (jump_dist_m > 15.0) {
            std::cout << "Symmetry Flip / Teleportation detected, rejecting AI matrix" << std::endl;
            candidate_valid = false;
          }
        }
      }

      used_fresh_matrix = true;
      last_good_matrix = H_candidate.clone();
      has_last_good_matrix = true;

      if (candidate_valid && smoothed_matrix.empty()) {
        smoothed_matrix = H_candidate.clone();
      } else if (candidate_valid) {
        // 70% new frame, 30% EMA history.
        cv::addWeighted(H_candidate, 0.7, smoothed_matrix, 0.3, 0.0, smoothed_matrix);
      }

      if (candidate_valid) {
        used_mode = "fresh_ai_ema";

        // Update Lucas-Kanade state from the current video frame.
        prev_gray = current_gray.clone();
        prev_pts.clear();
        // Anti-drift: spread features out across the frame.
        cv::goodFeaturesToTrack(prev_gray, prev_pts, 100, 0.01, 75);
      }
    } else {
      used_fresh_matrix = false;

      // If AI failed, use LK optical flow to update smoothed_matrix.
      if (!smoothed_matrix.empty() && !prev_gray.empty() && !prev_pts.empty()) {
        vector<cv::Point2f> curr_pts;
        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(prev_gray, current_gray, prev_pts, curr_pts, status, err);

        vector<cv::Point2f> prev_pts_f;
        vector<cv::Point2f> curr_pts_f;
        prev_pts_f.reserve(prev_pts.size());
        curr_pts_f.reserve(curr_pts.size());

        for (size_t k = 0; k < prev_pts.size() && k < curr_pts.size(); k++) {
          if (status.size() > k && status[k] == 1) {
            // Keep only successfully tracked points.
            prev_pts_f.push_back(prev_pts[k]);
            curr_pts_f.push_back(curr_pts[k]);
          }
        }

        if (prev_pts_f.size() >= 4) {
          vector<cv::Point2f> src_flow = prev_pts_f;
          vector<cv::Point2f> dst_flow = curr_pts_f;
          cv::Mat H_f2f = cv::findHomography(src_flow, dst_flow, cv::RANSAC);
          if (!H_f2f.empty() && H_f2f.rows == 3 && H_f2f.cols == 3) {
            // `smoothed_matrix` maps Image -> Pitch.
            // `H_f2f` maps Prev_Image -> Curr_Image.
            // To update the absolute homography for the current frame, we must
            // map backwards through the flow: Curr_Image -> Prev_Image -> Pitch.
            cv::Mat H_f2f_inv = H_f2f.inv();
            if (!H_f2f_inv.empty() && H_f2f_inv.rows == 3 && H_f2f_inv.cols == 3) {
              smoothed_matrix = smoothed_matrix * H_f2f_inv;
              used_mode = "optflow_ai_blindspot";
            }
          }
        }

        // Update tracking state for the next iteration regardless.
        prev_gray = current_gray.clone();
        prev_pts = curr_pts_f;
      } else {
        // No LK state available -> reuse last smoothed matrix as-is.
        used_mode = "smoothed_fallback";
      }
    }

    if (sample_logged < sample_limit) {
      cout << "Sample frame_idx=" << f0["frame_idx"]
           << ", valid=" << (candidate_valid ? 1 : 0)
           << ", cond=" << condNum
           << ", avgReprojErr=" << avgErr
           << ", used=" << used_mode
           << " (src pts=" << src.size() << ")" << endl;
      sample_logged++;
    }

    // Extra verification log around the historical failure window.
    const long long f0_idx = f0.contains("frame_idx") ? f0["frame_idx"].get<long long>() : -1LL;
    if (f0_idx >= 70 && f0_idx <= 150) {
      cout << "TrackVerify frame_idx=" << f0_idx
           << " valid=" << (candidate_valid ? 1 : 0)
           << " used=" << used_mode
           << " cond=" << condNum
           << " avgReprojErr=" << avgErr
           << " (LK prev_pts=" << prev_pts.size() << ")" << endl;
    }

    // Export homography for this frame (fresh AI or optical flow).
    if (!smoothed_matrix.empty()) {
      const long long frame_out_idx = f0.contains("frame_idx") ? f0["frame_idx"].get<long long>() : -1LL;
      if (frame_out_idx >= 0) {
        cv::Mat export_H = smoothed_matrix.inv(); // pitch->image
        if (!export_H.empty()) {
          cv::Mat export_H64;
          export_H.convertTo(export_H64, CV_64F);

          vector<vector<double>> h(3, vector<double>(3, 0.0));
          for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
              h[r][c] = export_H64.at<double>(r, c);
            }
          }

          output_json["homographies"].push_back(
              {
                  {"frame", frame_out_idx},
                  {"homography", h},
              });
        }
      }
    }
  }

  if (!smoothed_matrix.empty()) {
    cout << "Final smoothed homography:\n" << smoothed_matrix << "\n";
  }

  // Write output JSON for the Python tracker.
  const string outPath = "backend/output/match_test_homographies.json";
  try {
    std::filesystem::path p(outPath);
    std::filesystem::path parent = p.parent_path();
    if (!std::filesystem::exists(parent)) {
      std::filesystem::create_directories(parent);
    }
  } catch (...) {
    // Best-effort only.
  }

  ofstream out(outPath);
  if (!out.is_open()) {
    cerr << "Failed to open output JSON for writing: " << outPath << endl;
    return 1;
  }
  out << output_json.dump(2);
  out.close();

  return 0;
}

