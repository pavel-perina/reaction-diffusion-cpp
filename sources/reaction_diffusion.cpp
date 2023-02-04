// https://www.karlsims.com/rd.html
// https://www.youtube.com/watch?v=Iigfe7ZQfyY
#include <iostream>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <utility>

constexpr int N = 512;
constexpr float dx = 1.0f / N;
constexpr float dt = /*0.1f * dx * dx*/ 0.2;



class Update
{
public:
    Update(const cv::Mat& _u, const cv::Mat& _v, 
           cv::Mat& _uNext, cv::Mat& _vNext, 
           float _dx) 
        : u(_u)
        , v(_v)
        , uNext(_uNext)
        , vNext(_vNext)
        , dx(_dx)
    {
    }

    void operator() (const tbb::blocked_range<int> &range) const 
    {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = 1; j < N-1; j++) {
                // TODO: replace with [0.05 0.2 0.05   0.2 -1 0.2   0.05 0.2 0.05 kernel
                const float lap_u = u.at<float>(i-1, j) + u.at<float>(i+1, j) + u.at<float>(i, j-1) + u.at<float>(i, j+1) - 4.0*u.at<float>(i, j);
                const float lap_v = v.at<float>(i-1, j) + v.at<float>(i+1, j) + v.at<float>(i, j-1) + v.at<float>(i, j+1) - 4.0*v.at<float>(i, j);

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
    static constexpr float Du = 1.0f;
    static constexpr float Dv = 0.5f;
    static constexpr float f = 0.055f; // feed
    static constexpr float k = 0.062f; // kill

    const cv::Mat &u, &v;
    cv::Mat &uNext, &vNext;
    float dx;
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
   
    for (int i = 0; i < 30000; i++) {
        std::cout << "Loop " << i << std::endl;
        double minU, maxU, minV, maxV;
        cv::minMaxIdx(u, &minU, &maxU);
        cv::minMaxIdx(v, &minV, &maxV);
        //std::cout << "uMin= " << minU << ", uMax=" << maxU << std::endl;
        //std::cout << "vMin= " << minV << ", vMax=" << maxV << std::endl;


        tbb::parallel_for(tbb::blocked_range<int>(1, N-1), Update(u, v, uNext, vNext, dx));
        /*u = uNext.clone();
        v = vNext.clone();*/
        std::swap(u, uNext);
        std::swap(v, vNext);

    }

    {
        //cv::Mat crop = u.su
        double minU, maxU, minV, maxV;
        cv::minMaxIdx(u, &minU, &maxU);
        const auto [alpha, beta] = getGainOffset(minU, maxU);
        cv::Mat tmp;
        u.convertTo(tmp, CV_8UC1, alpha * 255.0, beta * 255.0);


        //cv::imwrite("result.png", 255 * (u - u.min()) / (u.max() - u.min()));
        cv::imwrite("result_u.webp", tmp);
    }
}
