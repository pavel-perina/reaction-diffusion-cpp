// https://www.karlsims.com/rd.html
// https://www.youtube.com/watch?v=Iigfe7ZQfyY
// video:
//  C:\apps\ffmpeg.exe -r 60  -i frame_%d.png -c:v libx264 -pix_fmt yuv420p out2.mp4
//  C:\apps\ffmpeg.exe -r 60  -i frame_%d.png -pix_fmt yuv420p out2.yuv
//  C:\apps\SvtAv1EncApp.exe -i .\out2.yuv -w 1280 -h 720 --fps-num 60000 --fps-denom 1001 -b out2.ivf
//  :\apps\mkvtoolnix\mkvmerge.exe .\out2.ivf -o reaction-diffusion.webm
#include <iostream>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <utility>
#include <sstream>

constexpr int W = 640;
constexpr int H = 480;

class UpdateBase 
{
protected:
    UpdateBase(const cv::Mat& _u, const cv::Mat& _v, cv::Mat& _uNext, cv::Mat& _vNext)
        : u(_u)
        , v(_v)
        , uNext(_uNext)
        , vNext(_vNext)
    {
    }

    static constexpr float Du = 0.21f;
    static constexpr float Dv = 0.105f;
    static constexpr float f = 0.055f; // feed
    static constexpr float k = 0.062f; // kill

    const cv::Mat& u, & v;
    cv::Mat& uNext, & vNext;
};

class Update 
    : public UpdateBase
{
public:
    Update(const cv::Mat& _u, const cv::Mat& _v, cv::Mat& _uNext, cv::Mat& _vNext)
        : UpdateBase(_u, _v, _uNext, _vNext)
    {
    }

    void operator() (const tbb::blocked_range<int> &range) const 
    {
        for (int i = range.begin(); i < range.end(); ++i) {
            const float* pU = u.ptr<float>(i);
            const float* pV = v.ptr<float>(i);
            const size_t rowSize = u.step / sizeof(float);

            const ptrdiff_t up   = (i > 0) 
                ? -rowSize
                : (H - 1)*rowSize;
            const ptrdiff_t down = (i + 1) < H
                ? rowSize
                : -(H - 1) * rowSize;

            for (int j = 0; j < W; j++) {
                // TODO: replace with [0.05 0.2 0.05   0.2 -1 0.2   0.05 0.2 0.05 kernel
                float lap_u, lap_v; 
                if (j == 0) { // unlikely
                    lap_u = pU[up] + pU[down] + u.at<float>(i, W - 1) + pU[+1] - 4.0f * pU[0];
                    lap_v = pV[up] + pV[down] + v.at<float>(i, W - 1) + pV[+1] - 4.0f * pV[0];
                }
                else if (j == W - 1) { // unlikely
                    lap_u = pU[up] + pU[down] + pU[-1] + u.at<float>(i, 0) - 4.0f * pU[0];
                    lap_v = pV[up] + pV[down] + pV[-1] + v.at<float>(i, 0) - 4.0f * pV[0];

                }
                else { // likely
                    lap_u = pU[down] + pU[up] + pU[-1] + pU[+1] - 4.0f * pU[0];
                    lap_v = pV[down] + pV[up] + pV[-1] + pV[+1] - 4.0f * pV[0];
                }

                const float _u = u.at<float>(i, j);
                const float _v = v.at<float>(i, j);
                const float _uNext = _u + (Du * lap_u - _u * _v * _v + f * (1.0f - _u));
                const float _vNext = _v + (Dv * lap_v + _u * _v * _v - (f + k) * _v);
                uNext.at<float>(i, j) = _uNext;
                vNext.at<float>(i, j) = _vNext;
                pU++;
                pV++;

            }
        }
    }
};




std::pair<double, double> getGainOffset(double minVal, double maxVal)
{
    const double gain   = (maxVal > minVal) ? 1.0 / (maxVal - minVal) : 1.0;
    const double offset = -minVal * gain;
    return { gain, offset };
}


int main() 
{
    cv::Mat u     = cv::Mat::ones(H, W, CV_32FC1);
    cv::Mat v     = cv::Mat::zeros(H, W, CV_32FC1);
    cv::Mat uNext = cv::Mat::ones(H, W, CV_32FC1);
    cv::Mat vNext = cv::Mat::zeros(H, W, CV_32FC1);

    // Initialize image with rectange (random numbers do not work)
    std::cout << "Initializing arrays\n";
    {
        auto seed = std::random_device{}();
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution uniDist{ 0.0, 0.99 };
        for (int i = H/2-20; i < H/2+20; ++i) {
            for (int j = W/2-10; j < W/2+10; ++j) {
#if 0
                v.at<float>(i, j) = 1.0;
#else
                v.at<float>(i, j) = (float)uniDist(rng);
                u.at<float>(i, j) = (float)uniDist(rng);
#endif
            }
        }
    }
    
    // This is there only to slow down video at start and accelerate it in the end
    constexpr int maxIter = 64000;
    std::vector<int> framesToSave;
    {
        const int nFramesToSave = 60 * 8 - 1; // 60fps, 8s
        framesToSave.reserve(nFramesToSave);
        framesToSave.emplace_back(0);
        double gamma = 3.0;
        for (int i = 1; i < nFramesToSave; ++i) {
            double timeLinePos = (double)i / nFramesToSave;
            int frame = (int)(pow(timeLinePos, gamma) * maxIter);
            if (framesToSave.back() != frame) {
                framesToSave.emplace_back(frame);
            }
        }
    }

    int j = 0;
    for (int i = 0; i < maxIter; i++) {
        tbb::parallel_for(tbb::blocked_range<int>(0, H), Update(u, v, uNext, vNext));
        std::swap(u, uNext);
        std::swap(v, vNext);

        // save video frame
        if (i == framesToSave[j]) {
            double minU, maxU;
            cv::minMaxIdx(u, &minU, &maxU);
            const auto [alpha, beta] = getGainOffset(minU, maxU);
            cv::Mat tmp;
            u.convertTo(tmp, CV_8UC1, alpha * 255.0, beta * 255.0);

            std::ostringstream oss;
            std::cout << "Loop " << i << std::endl;
            oss << "frame_" << j << ".png";
            cv::imwrite(oss.str(), tmp);
            j++;
        }
    }
}
