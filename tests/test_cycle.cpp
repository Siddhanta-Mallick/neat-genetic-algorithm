#include "Genotype.hpp"
#include "Utils.hpp"
#include <../external/catch.hpp>
#include <vector>

TEST_CASE("Simple DAG - no cycle") {
  std::vector<LinkGene> links = {
      {LinkGene{{-1, 0}, 0, true}},
      {LinkGene{{-2, 0}, 0, true}},
      {LinkGene{{-3, 1}, 0, true}},
      {LinkGene{{-4, 1}, 0, true}},
  };

  REQUIRE(CycleDetector::would_contain_cycle(links, 0, 1, 6, 4) == false);
}

TEST_CASE("Adding edge creates cycle")
{
  std::vector<LinkGene> links = {
      {LinkGene{{-1, 1}, 0, true}},
      {LinkGene{{-1, 4}, 0, true}},
      {LinkGene{{1, 2}, 0, true}},
      {LinkGene{{2, 0}, 0, true}},
      {LinkGene{{3, 0}, 0, true}},
      {LinkGene{{3, 1}, 0, true}},
      {LinkGene{{4, 3}, 0, true}}
  };

  REQUIRE(CycleDetector::would_contain_cycle(links, 2, 3, 6, 1) == true);
}

TEST_CASE("Adding edge creates long cycle")
{
  
  std::vector<LinkGene> links = {
      {LinkGene{{-1, 1}, 0, true}},
      {LinkGene{{1, 2}, 0, true}},
      {LinkGene{{2, 3}, 0, true}},
      {LinkGene{{3, 4}, 0, true}},
      {LinkGene{{4, 5}, 0, true}},
      {LinkGene{{5, 6}, 0, true}},
      {LinkGene{{6, 0}, 0, true}}
  };

  REQUIRE(CycleDetector::would_contain_cycle(links, 6, 1, 8, 1) == true);
}


TEST_CASE("Adding link to cause self loop creates cycle")
{
  
  std::vector<LinkGene> links = {
      {LinkGene{{-1, 0}, 0, true}},
      {LinkGene{{-2, 0}, 0, true}},
      {LinkGene{{-3, 1}, 0, true}},
      {LinkGene{{-4, 1}, 0, true}},
  };

  REQUIRE(CycleDetector::would_contain_cycle(links, 0, 0, 6, 4) == true);
}
