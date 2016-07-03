#include "sgd_solver.hpp"
#include "common.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <ml/include/ml.hpp>
#include <io/general_fstream.hpp>

namespace softmax {

SGDSolver::SGDSolver(const SGDSolverConfig& config) :
    w_table_(config.w_table), feature_dim_(config.feature_dim),
    w_table_num_cols_(config.w_table_num_cols),
    num_labels_(config.num_labels), w_dim_(feature_dim_ * num_labels_),
    lambda_(config.lambda), cost_sensitive_(config.cost_sensitive) {
    w_cache_.resize(num_labels_);
    w_cache_old_.resize(num_labels_);
    for (int i = 0; i < num_labels_; ++i) {
        w_cache_[i] = petuum::ml::DenseFeature<float>(feature_dim_);
        w_cache_old_[i] = std::vector<float>(feature_dim_);
    }
    
    if (cost_sensitive_) {
		CreateCostMatrix();
	}

    if (config.sparse_data) {
        FeatureDotProductFun_ = petuum::ml::SparseDenseFeatureDotProduct;
    } else {
        FeatureDotProductFun_ = petuum::ml::DenseDenseFeatureDotProduct;
    }
}

void SGDSolver::CreateCostMatrix() {
	std::vector<std::vector<float>> costMat(num_labels_);
	for (int i = 0; i < num_labels_; ++i) {
		std::vector<float> costMatRow(num_labels_);
		
		int rightEnd = num_labels_ - i;
		
		for (int r = 0; r < rightEnd; ++r) {
			costMatRow[r + i] = static_cast<float>(r);
		}
		
		int leftStart = i - 1;
		
		int val = 1;
		for (int l = leftStart; l >= 0; --l) {
			costMatRow[l] = static_cast<float>(val);
			val++;
		}
		
		costMat[i] = costMatRow;
	}
	
	cost_mat_ = costMat;
}

void SGDSolver::RefreshParams() {
    // Write delta's to PS table.
    int num_full_rows = feature_dim_ / w_table_num_cols_;
    int num_rows_per_label = std::ceil(static_cast<float>(feature_dim_)
                                       / w_table_num_cols_);
    for (int l = 0; l < num_labels_; ++l) {
        std::vector<float> w_delta(feature_dim_);
        std::vector<float>& w_cache_vec = w_cache_[l].GetVector();
        for (int j = 0; j < feature_dim_; ++j) {
            w_delta[j] = w_cache_vec[j] - w_cache_old_[l][j];
            CHECK(!std::isnan(w_delta[j])) << "nan detected.";
        }
        // Write delta's to PS table.
        for (int k = 0; k < num_full_rows; ++k) {
            petuum::UpdateBatch<float> w_update_batch(w_table_num_cols_);
            for (int j = 0; j < w_table_num_cols_; ++j) {
                int idx = k * w_table_num_cols_ + j;
                w_update_batch.UpdateSet(j, j, w_delta[idx]);
            }
            w_table_.BatchInc(num_rows_per_label * l + k, w_update_batch);
        }

        // last incomplete row.
        if (feature_dim_ % w_table_num_cols_ != 0) {
            int num_cols_last_row = feature_dim_ - num_full_rows * w_table_num_cols_;
            petuum::UpdateBatch<float> w_update_batch(num_cols_last_row);
            for (int j = 0; j < num_cols_last_row; ++j) {
                int idx = num_full_rows * w_table_num_cols_ + j;
                w_update_batch.UpdateSet(j, j, w_delta[idx]);
            }
            w_table_.BatchInc(num_rows_per_label * l + num_full_rows,
                              w_update_batch);
        }

        // Read w from the PS.
        std::vector<float> w_cache(w_table_num_cols_);
        for (int k = 0; k < num_full_rows; ++k) {
            petuum::RowAccessor row_acc;
            const auto& r = w_table_.Get<petuum::DenseRow<float>>(
                                num_rows_per_label * l + k, &row_acc);
            r.CopyToVector(&w_cache);
            std::copy(w_cache.begin(), w_cache.end(),
                      w_cache_vec.begin() + k * w_table_num_cols_);
        }
        if (feature_dim_ % w_table_num_cols_ != 0) {
            // last incomplete row.
            int num_cols_last_row = feature_dim_ - num_full_rows * w_table_num_cols_;
            petuum::RowAccessor row_acc;
            const auto& r = w_table_.Get<petuum::DenseRow<float>>(
                                num_rows_per_label * l + num_full_rows, &row_acc);
            r.CopyToVector(&w_cache);
            std::copy(w_cache.begin(), w_cache.begin() + num_cols_last_row,
                      w_cache_vec.begin() + num_full_rows * w_table_num_cols_);
        }
        w_cache_old_[l] = w_cache_vec;
    }
}

int32_t SGDSolver::ZeroOneLoss(const std::vector<float>& prediction,
                               int32_t label) const {
    float max_val = prediction[0];
    int32_t max_idx = 0;
    for (int i = 1; i < num_labels_; ++i) {
        if (prediction[i] > max_val) {
            max_val = prediction[i];
            max_idx = i;
        }
    }
    return (max_idx == label) ? 0 : 1;
}

int32_t SGDSolver::MPLabel(const std::vector<float>& prediction) const {
    float max_val = prediction[0];
    int32_t max_idx = 0;
    
    if (cost_sensitive_) {
		max_idx = CostSensitivePrediction(prediction);
	}
	else {
		for (int i = 1; i < num_labels_; ++i) {
			if (prediction[i] > max_val) {
				max_val = prediction[i];
				max_idx = i;
			}
		}
	}
    
    return max_idx;
}


float SGDSolver::CrossEntropyLoss(const std::vector<float>& prediction,
                                  int32_t label) const {
    CHECK_LE(prediction[label], 1);
    return -petuum::ml::SafeLog(prediction[label]);
}


std::vector<float> SGDSolver::Predict(
    const petuum::ml::AbstractFeature<float>& feature) const {
    std::vector<float> y_vec(num_labels_);
    for (int i = 0; i < num_labels_; ++i) {
        y_vec[i] = FeatureDotProductFun_(feature, w_cache_[i]);
    }
    petuum::ml::Softmax(&y_vec);
    return y_vec;
}

int32_t SGDSolver::CostSensitivePrediction(const std::vector<float>& prediction) const {
	
	std::vector<float> costPrediction;
	
	for (int i = 0; i < num_labels_; ++i) {
		std::vector<float> costVector = cost_mat_[i];
		
		float sum = 0.0f;
		for (int k = 0; k < num_labels_; ++k) {
			sum += prediction[k] * costVector[k];
		}
		
		costPrediction.push_back(sum);
	}
	// get the index (label) of the element with smallest cost
	int label = std::min_element(costPrediction.begin(), costPrediction.end()) - costPrediction.begin();
	
	return label;
}

void SGDSolver::MiniBatchSGD(
    const std::vector<petuum::ml::AbstractFeature<float>*>& features,
    const std::vector<int32_t>& labels,
    const std::vector<int32_t>& idx, double lr) {
    for (const auto& i : idx) {
        petuum::ml::AbstractFeature<float>& feature = *features[i];
        int32_t label = labels[i];
        std::vector<float> y_vec = Predict(feature);
        y_vec[label] -= 1.; // See Bishop PRML (2006) Eq. (4.109)

        // outer product
        for (int i = 0; i < num_labels_; ++i) {
            // w_cache_[i] += -\eta * y_vec[i] * feature
            petuum::ml::FeatureScaleAndAdd(-lr * y_vec[i], feature,
                                           &w_cache_[i]);
        }

    }

    for (int i = 0; i < num_labels_; ++i) {
        ApplyL2Regularization(i, lr);
    }
}

void SGDSolver::SaveWeights(const std::string& filename) const {
    petuum::io::ofstream w_stream(filename,
                                  std::ofstream::out | std::ofstream::trunc);
    CHECK(w_stream);
    // Print meta data
    w_stream << "num_labels: " << num_labels_ << std::endl;
    w_stream << "feature_dim: " << feature_dim_ << std::endl;

    for (int i = 0; i < num_labels_; ++i) {
        int num_entries = w_cache_[i].GetNumEntries();
        for (int j = 0; j < num_entries; ++j) {
            int feature_id = w_cache_[i].GetFeatureId(j);
            float feature_val = w_cache_[i].GetFeatureVal(j);
            w_stream << feature_id << ":" << feature_val << " ";
        }
        w_stream << std::endl;
    }
    w_stream.close();
    LOG(INFO) << "Saved weight to " << filename;
}

void SGDSolver::ApplyL2Regularization(int label, double lr) {

    if (lambda_ > 0)
    {
        for (int i = 0; i < w_cache_[label].GetNumEntries(); ++i) {
            int32_t fid = w_cache_[label].GetFeatureId(i);
            float f_val = w_cache_[label].GetFeatureVal(i);
            w_cache_[label].SetFeatureVal(fid, f_val + lambda_ * lr * f_val);
        }
    }

}

}
