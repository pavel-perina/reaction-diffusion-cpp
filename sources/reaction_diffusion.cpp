// https://www.karlsims.com/rd.html
// https://www.youtube.com/watch?v=Iigfe7ZQfyY
// video:
//  C:\apps\ffmpeg.exe -r 60  -i frame_%d.png -c:v libx264 -pix_fmt yuv420p out2.mp4
//  C:\apps\ffmpeg.exe -r 60  -i frame_%d.png -pix_fmt yuv420p out2.yuv
//  C:\apps\SvtAv1EncApp.exe -i .\out2.yuv -w 1280 -h 720 --fps-num 60000 --fps-denom 1001 -b out2.ivf
//  :\apps\mkvtoolnix\mkvmerge.exe .\out2.ivf -o reaction-diffusion.webm

#include "alignment_allocator.h"

#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <utility>

#include <tbb/tbb.h>

#include <fmt/core.h>
#include <fmt/chrono.h>

#include <opencv2/opencv.hpp>

#include <immintrin.h>

constexpr int W = 512;
constexpr int H = 512;

class IUpdater 
{
public:
    virtual ~IUpdater() = default;
    virtual void iterate() = 0;
};



class UpdaterBase
    : public IUpdater
{
protected:
    UpdaterBase(cv::Mat& _u, cv::Mat& _v, cv::Mat& _uNext, cv::Mat& _vNext)
        : u(_u)
        , v(_v)
        , uNext(_uNext)
        , vNext(_vNext)
    {
    }

    static constexpr float Du = 0.21f;
    static constexpr float Dv = 0.105f;
    static constexpr float f  = 0.055f; // feed
    static constexpr float k  = 0.062f; // kill

    cv::Mat& u, & v;
    cv::Mat& uNext, & vNext;
};


class Updater1
    : public UpdaterBase
{
public:
    Updater1(cv::Mat& _u, cv::Mat& _v, cv::Mat& _uNext, cv::Mat& _vNext)
        : UpdaterBase(_u, _v, _uNext, _vNext)
    {
    }

    void operator() (const tbb::blocked_range<int>& range) const
    {
        for (int i = range.begin(); i < range.end(); ++i) {
            const auto iUp = (i > 0) ? i - 1 : H - 1;
            const auto iDown = (i + 1) % H;
            for (int j = 0; j < W; j++) {
                // TODO: replace with [0.05 0.2 0.05   0.2 -1 0.2   0.05 0.2 0.05 kernel
                float lap_u, lap_v;
                if (j == 0) { // unlikely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, W - 1) + u.at<float>(i, j + 1) - 4.0f * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, W - 1) + v.at<float>(i, j + 1) - 4.0f * v.at<float>(i, j);
                }
                else if (j == W - 1) { // unlikely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, j - 1) + u.at<float>(i, 0) - 4.0f * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, j - 1) + v.at<float>(i, 0) - 4.0f * v.at<float>(i, j);

                }
                else { // likely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, j - 1) + u.at<float>(i, j + 1) - 4.0f * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, j - 1) + v.at<float>(i, j + 1) - 4.0f * v.at<float>(i, j);
                }

                const float _u = u.at<float>(i, j);
                const float _v = v.at<float>(i, j);
                const float _uNext = _u + (Du * lap_u - _u * _v * _v + f * (1.0f - _u));
                const float _vNext = _v + (Dv * lap_v + _u * _v * _v - (f + k) * _v);
                uNext.at<float>(i, j) = _uNext;
                vNext.at<float>(i, j) = _vNext;
            }
        }
    }

    void iterate() override
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, H), [&](const tbb::blocked_range<int>& range) {
            this->operator()(range);
        });
        std::swap(u, uNext);
        std::swap(v, vNext);
    }

};



class Updater2
    : public UpdaterBase
{
public:
    Updater2(cv::Mat& _u, cv::Mat& _v, cv::Mat& _uNext, cv::Mat& _vNext)
        : UpdaterBase(_u, _v, _uNext, _vNext)
    {
    }

    void operator() (const tbb::blocked_range<int> &range) const 
    {
        for (int i = range.begin(); i < range.end(); ++i) {
            const float* pU = u.ptr<float>(i);
            const float* pV = v.ptr<float>(i);
            const size_t rowSize = u.step / sizeof(float);

            const ptrdiff_t up   = (i > 0) 
                ? -(ptrdiff_t)rowSize
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

    void iterate() override
    {
        tbb::parallel_for(
            tbb::blocked_range<int>(0, H), 
            [&](const tbb::blocked_range<int>& range) {
                this->operator()(range);
            });
        std::swap(u, uNext);
        std::swap(v, vNext);
    }
};



class Updater3
    : public UpdaterBase
{
public:
    Updater3(cv::Mat & _u, cv::Mat & _v, cv::Mat & _uNext, cv::Mat & _vNext)
        : UpdaterBase(_u, _v, _uNext, _vNext)
    {
    }

    void processArea(int x1, int y1, int x2, int y2) 
    {
        for (int i = y1; i < y2; ++i) {
            const auto iUp   = (i > 0) ? i - 1 : H - 1;
            const auto iDown = (i + 1) % H;
            for (int j = x1; j < x2; ++j) {
                // TODO: replace with [0.05 0.2 0.05   0.2 -1 0.2   0.05 0.2 0.05 kernel
                float lap_u, lap_v;
                if (j == 0) { // unlikely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, W - 1) + u.at<float>(i, j + 1) - 4.0f * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, W - 1) + v.at<float>(i, j + 1) - 4.0f * v.at<float>(i, j);
                }
                else if (j == W - 1) { // unlikely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, j - 1) + u.at<float>(i, 0) - 4.0f * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, j - 1) + v.at<float>(i, 0) - 4.0f * v.at<float>(i, j);

                }
                else { // likely
                    lap_u = u.at<float>(iUp, j) + u.at<float>(iDown, j) + u.at<float>(i, j - 1) + u.at<float>(i, j + 1) - 4.0f * u.at<float>(i, j);
                    lap_v = v.at<float>(iUp, j) + v.at<float>(iDown, j) + v.at<float>(i, j - 1) + v.at<float>(i, j + 1) - 4.0f * v.at<float>(i, j);
                }
                const float _u = u.at<float>(i, j);
                const float _v = v.at<float>(i, j);
                const float _uNext = _u + (Du * lap_u - _u * _v * _v + f * (1.0f - _u));
                const float _vNext = _v + (Dv * lap_v + _u * _v * _v - (f + k) * _v);
                uNext.at<float>(i, j) = _uNext;
                vNext.at<float>(i, j) = _vNext;
            }
        }
    }

    constexpr int leftStop() const
    {
        return 8; // 8 floats are 32bytes for AVX alignment
    }

    int rightStop() 
    {
        return ((W - 1) / 8) * 8;
    }

    void processBorders()
    {
        // FIXME: can possibly crash on small image
        static_assert(W > 8, "Image to small");
        processArea(0,     0, W, 1);     // top
        processArea(0, H - 1, W, H);     // bottom
        processArea(0,     1, leftStop(), H - 1); // left
        processArea(rightStop(), 1, W, H - 1); // right

    }

    void processInside()
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, H), [&](tbb::blocked_range<int>& range) {
            const int jMin = leftStop();
            const int jMax = rightStop();
            const __m256 f_      = _mm256_set1_ps(f);
            const __m256 negKF      = _mm256_set1_ps(-(k+f));
            const __m256 one     = _mm256_set1_ps(1.0f);
            const __m256 negFour = _mm256_set1_ps(-4.0f);
            const __m256 Du_     = _mm256_set1_ps(Du);
            const __m256 Dv_     = _mm256_set1_ps(Dv);

            for (int i = range.begin(); i < range.end(); ++i) {
                const auto iUp   = (i > 0) ? i - 1 : H - 1;
                const auto iDown = (i + 1) % H;
                for (int j = jMin; j < jMax; j += 8) {

                    __m256 uCurrent = _mm256_load_ps(&u.at<float>(i, j));

                    __m256 uUp      = _mm256_load_ps (&u.at<float>(iUp, j));
                    __m256 uDown    = _mm256_load_ps (&u.at<float>(iDown, j));
                    __m256 uLeft    = _mm256_loadu_ps(&u.at<float>(i, j - 1));
                    __m256 uRight   = _mm256_loadu_ps(&u.at<float>(i, j + 1));
 
                    __m256 lap_u = _mm256_add_ps(uUp, uDown);
                    lap_u = _mm256_add_ps(lap_u, uLeft);
                    lap_u = _mm256_add_ps(lap_u, uRight);
                    lap_u = _mm256_add_ps(lap_u, _mm256_mul_ps(uCurrent, negFour));
                    __m256 Du_lap_u = _mm256_mul_ps(Du_, lap_u);

                    // f*(1-u)
                    __m256 fTerm = _mm256_sub_ps(one, uCurrent);
                    fTerm = _mm256_mul_ps(f_, fTerm);

                    __m256 vCurrent = _mm256_load_ps(&v.at<float>(i, j));
                    __m256 uvv = _mm256_mul_ps(uCurrent, _mm256_mul_ps(vCurrent, vCurrent));

                    //  _u + (Du * lap_u - _u * _v * _v + f * (1.0f - _u));
                    __m256 uNext1 = _mm256_sub_ps(fTerm, uvv);
                    __m256 uNext2 = _mm256_add_ps(uCurrent, Du_lap_u);
                    __m256 uNext_ = _mm256_add_ps(uNext1, uNext2);
                    _mm256_store_ps(&uNext.at<float>(i, j), uNext_);

                    __m256 vUp    = _mm256_load_ps (&v.at<float>(iUp, j));
                    __m256 vDown  = _mm256_load_ps (&v.at<float>(iDown, j));
                    __m256 vLeft  = _mm256_loadu_ps(&v.at<float>(i, j - 1));
                    __m256 vRight = _mm256_loadu_ps(&v.at<float>(i, j + 1));

                    __m256 lap_v = _mm256_add_ps(vUp, vDown);
                    lap_v = _mm256_add_ps(lap_v, vLeft);
                    lap_v = _mm256_add_ps(lap_v, vRight);
                    lap_v = _mm256_add_ps(lap_v, _mm256_mul_ps(vCurrent, negFour));
                    __m256 Dv_lap_v = _mm256_mul_ps(Dv_, lap_v);

                    __m256 kTerm  = _mm256_mul_ps(negKF, vCurrent);
                    __m256 vNext1 = _mm256_add_ps(kTerm, uvv);
                    __m256 vNext2 = _mm256_add_ps(vCurrent, Dv_lap_v);
                    __m256 vNext_ = _mm256_add_ps(vNext1, vNext2);
                    _mm256_store_ps(&vNext.at<float>(i, j), vNext_);
                }
            }
        } // , tbb::*partitioner()
        );
    }

    void iterate() override
    {
        processBorders();
        processInside();
        std::swap(u, uNext);
        std::swap(v, vNext);
    }
};



std::pair<double, double> getGainOffset(double minVal, double maxVal)
{
    const double gain   = (maxVal > minVal) ? 1.0 / (maxVal - minVal) : 1.0;
    const double offset = -minVal * gain;
    return { gain, offset };
}



void saveImage(const std::string& filename, const cv::Mat& m)
{
    double minVal, maxVal;
    cv::minMaxIdx(m, &minVal, &maxVal);
    const auto [alpha, beta] = getGainOffset(minVal, maxVal);
    cv::Mat tmp;
    m.convertTo(tmp, CV_8UC1, alpha * 255.0, beta * 255.0);
    cv::imwrite(filename, tmp);
}



int main()
{
    int numThreads = oneapi::tbb::info::default_concurrency();
    fmt::print("Machine has {} threads.\n", numThreads);
    numThreads = 2;
    fmt::print("Limiting to {} threads.\n", numThreads);
    tbb::global_control tbbGlob(tbb::global_control::max_allowed_parallelism, numThreads);

    cv::Mat u, v, uNext, vNext;
    // would be nice to use aligned alocator for unique_ptr<float[]>
    std::array<std::vector<float, AlignmentAllocator<float, 4096>>, 4> memoryBlocks;
    {
        constexpr size_t padTo = 64;
        size_t stride = ((W * sizeof(float) + (padTo - 1)) / padTo) * padTo;
        // we need padding to 32 bytes or 8 floats
        for (auto &block : memoryBlocks) {
            block.resize(H * stride / sizeof(float));
        }
        u     = cv::Mat(H, W, CV_32FC1, memoryBlocks[0].data(), stride);
        v     = cv::Mat(H, W, CV_32FC1, memoryBlocks[1].data(), stride);
        uNext = cv::Mat(H, W, CV_32FC1, memoryBlocks[2].data(), stride);
        vNext = cv::Mat(H, W, CV_32FC1, memoryBlocks[3].data(), stride);
        u.setTo(1.0f);
    }

    // Initialize image with rectange (random numbers do not work)
    fmt::print("Initializing arrays\n");
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

#if 0
    int j = 0;
    for (int i = 0; i < maxIter; i++) {
        tbb::parallel_for(tbb::blocked_range<int>(0, H), Update(u, v, uNext, vNext));
        std::swap(u, uNext);
        std::swap(v, vNext);

        // save video frame
        if (i == framesToSave[j]) {
            fmt::print("Loop {:06}\n", i);
            saveImage(fmt::format("frame_{:06}.png", j), u);
            j++;
        }
    }
#else
    int testIter = 2000;
    // TODO: warm up and repeat, first test has different data
    // TODO: add functions for logical parts
    using Clock = std::chrono::system_clock;
    using DurationMs = std::chrono::duration<double, std::milli>;

    Updater1 updater1(u, v, uNext, vNext);
    Updater2 updater2(u, v, uNext, vNext);
    Updater3 updater3(u, v, uNext, vNext);
    for (int j = 0; j < 5; ++j) {
        Clock::time_point stopWatchStart = Clock::now();
        for (int i = 0; i < testIter; i++) {
            updater1.iterate();
        }
        DurationMs durationMs(Clock::now() - stopWatchStart);
        fmt::print("Duration: {:>8.3} {}\n", durationMs, "Updater1: trivial");
        // 
        stopWatchStart = Clock::now();
        for (int i = 0; i < testIter; i++) {
            updater2.iterate();
        }
        durationMs = Clock::now() - stopWatchStart;
        fmt::print("Duration: {:>8.3} {}\n", durationMs, "Updater2: no mat.at<float>(x,y)");

        // 
        stopWatchStart = Clock::now();
        for (int i = 0; i < testIter; i++) {
            updater3.iterate();
        }
        durationMs = Clock::now() - stopWatchStart;
        fmt::print("Duration: {:>8.3} {}\n", durationMs, "Updater3: AVX");


    }
#endif
}
