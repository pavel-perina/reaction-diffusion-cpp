// https://www.karlsims.com/rd.html
// https://www.youtube.com/watch?v=Iigfe7ZQfyY
#include <iostream>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <utility>
#include <sstream>

constexpr int N = 512;


class Update
{
public:
    Update(const cv::Mat& _u, const cv::Mat& _v, cv::Mat& _uNext, cv::Mat& _vNext) 
        : u(_u)
        , v(_v)
        , uNext(_uNext)
        , vNext(_vNext)
    {
    }

    void operator() (const tbb::blocked_range<int> &range) const 
    {
        for (int i = range.begin(); i < range.end(); ++i) {
            const auto iUp   = (i > 0) ? i - 1 : N - 1;
            const auto iDown = (i + 1) % N;
            for (int j = 0; j < N; j++) {
                // TODO: replace with [0.05 0.2 0.05   0.2 -1 0.2   0.05 0.2 0.05 kernel
                float lap_u, lap_v; 
                if (j == 0) { // unlikely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, N - 1) + u.at<float>(i, j + 1) - 4.0 * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, N - 1) + v.at<float>(i, j + 1) - 4.0 * v.at<float>(i, j);
                }
                else if (j == N - 1) { // unlikely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, j - 1) + u.at<float>(i,     0) - 4.0 * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, j - 1) + v.at<float>(i,     0) - 4.0 * v.at<float>(i, j);

                }
                else { // likely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, j - 1) + u.at<float>(i, j + 1) - 4.0 * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, j - 1) + v.at<float>(i, j + 1) - 4.0 * v.at<float>(i, j);
                }

                const float _u = u.at<float>(i, j);
                const float _v = v.at<float>(i, j);
                const float _uNext = _u + dt * (Du * lap_u - _u * _v * _v + f * (1.0 - _u));
                const float _vNext = _v + dt * (Dv * lap_v + _u * _v * _v - (f + k) * _v);
                uNext.at<float>(i, j) = _uNext;
                vNext.at<float>(i, j) = _vNext;

            }
        }
    }

private:
    static constexpr float Du = 0.21f;
    static constexpr float Dv = 0.105f;
    static constexpr float f = 0.055f; // feed
    static constexpr float k = 0.062f; // kill

    const cv::Mat &u, &v;
    cv::Mat &uNext, &vNext;
};


std::pair<double, double> getGainOffset(double minVal, double maxVal)
{
    const double gain   = (maxVal > minVal) ? 1.0 / (maxVal - minVal) : 1.0;
    const double offset = -minVal * gain;
    return { gain, offset };
}


int main() 
{
    cv::Mat u     = cv::Mat::ones(N, N, CV_32FC1);
    cv::Mat v     = cv::Mat::zeros(N, N, CV_32FC1);
    cv::Mat uNext = cv::Mat::ones(N, N, CV_32FC1);
    cv::Mat vNext = cv::Mat::zeros(N, N, CV_32FC1);

    std::cout << "Initializing arrays\n";
    {
        auto seed = std::random_device{}();
        std::mt19937_64 rng(seed);

#if 1
        for (int i = N/2-20; i < N/2+20; ++i) {
            for (int j = N/2-10; j < N/2+10; ++j) {
                v.at<float>(i, j) = 1.0;
            }
        }
#else
        std::uniform_real_distribution uniDist{ 0.01, 0.99 };
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N ; ++j) {
                u.at<float>(i, j) = uniDist(rng);
                v.at<float>(i, j) = uniDist(rng);
            }
        }
#endif
    }
   
    for (int i = 0; i < 256000; i++) {
        tbb::parallel_for(tbb::blocked_range<int>(0, N), Update(u, v, uNext, vNext));
        std::swap(u, uNext);
        std::swap(v, vNext);

        if (i % 512 == 0) {
            double minU, maxU, minV, maxV;
            cv::minMaxIdx(u, &minU, &maxU);
            const auto [alpha, beta] = getGainOffset(minU, maxU);
            cv::Mat tmp;
            u.convertTo(tmp, CV_8UC1, alpha * 255.0, beta * 255.0);

            std::ostringstream oss;
            std::cout << "Loop " << i << std::endl;
            oss << "frame_" << i << ".png";
            cv::imwrite(oss.str(), tmp);
        }
    }
}
