#include <iostream>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <random>

constexpr int N = 512;
constexpr float dx = 1.0f / N;
constexpr float dt = 0.1f * dx * dx;


class Update
{
public:
    Update(cv::Mat &_u, cv::Mat &_v, float _dx) 
        : u(_u)
        , v(_v)
        , dx(_dx)
    {        
    }

    void operator() (const tbb::blocked_range2d<int> &r) const 
    {
        for (int i = r.rows().begin(); i < r.rows().end(); i++) {
            for (int j = r.cols().begin(); j < r.cols().end(); j++) {
                const float lap_u = u.at<float>(i-1, j) + u.at<float>(i+1, j) + u.at<float>(i, j-1) + u.at<float>(i, j+1) - 4*u.at<float>(i, j);
                const float lap_v = v.at<float>(i-1, j) + v.at<float>(i+1, j) + v.at<float>(i, j-1) + v.at<float>(i, j+1) - 4*v.at<float>(i, j);

                u.at<float>(i, j) += dt * (Du * lap_u / (dx * dx) - u.at<float>(i, j) * v.at<float>(i, j) * v.at<float>(i, j) +  f      * (1 - u.at<float>(i, j)));
                v.at<float>(i, j) += dt * (Dv * lap_v / (dx * dx) + u.at<float>(i, j) * v.at<float>(i, j) * v.at<float>(i, j) - (f + k) *      v.at<float>(i, j));
            }
        }
    }

private:
    static constexpr float Du = 0.16f;
    static constexpr float Dv = 0.08f;
    static constexpr float f = 0.04f; // feed
    static constexpr float k = 0.06f; // kill
*/

    cv::Mat &u, &v;
    float dx;
};

int main() 
{
    std::cout << "Hello" << std::endl;
    cv::Mat u = cv::Mat::ones(N, N, CV_32FC1);
    cv::Mat v = cv::Mat::zeros(N, N, CV_32FC1);

    std::cout << "Initializing arrays\n";
    {
        auto seed = std::random_device{}();
        std::mt19937_64 rng(seed);

        std::normal_distribution<float> normDist{ 0.5f, 0.1f }; //mean followed by stdiv    
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                u.at<float>(i, j) = normDist(rng);
                v.at<float>(i, j) = normDist(rng);
            }
        }
    }
   
    for (int i = 0; i < 1000; i++) {
        std::cout << "Loop " << i << std::endl;
        tbb::parallel_for(tbb::blocked_range2d<int>(1, N-1, 1, N-1), Update(u, v, dx));
    }

    //cv::imwrite("result.png", 255 * (u - u.min()) / (u.max() - u.min()));
    cv::imwrite("result.png", u);
}
