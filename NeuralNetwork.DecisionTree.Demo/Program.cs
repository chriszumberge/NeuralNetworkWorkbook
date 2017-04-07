using NeuralNetwork.DecisionTree;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.DecisionTree.Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            //var fishSet = GetFishDataSet();

            //Tree tree = fishSet.BuildTree();

            //Console.WriteLine(tree.DisplayTree());

            //Console.WriteLine("Testing new data");

            //var testInstance = new Instance
            //{
            //    Features = new List<Feature>
            //    {
            //        new Feature("0", FISH_CAN_SURVIVE_WITHOUT_SURFACING),
            //        new Feature("1", FISH_HAS_FLIPPERS)
            //    }
            //};
            //var output = Tree.ProcessInstance(tree, testInstance);
            //Console.WriteLine($"{output.Axis}: {output.Value}");

            var heroSet = GetHeroDataSet();
            Tree tree = heroSet.BuildTree();
            Console.WriteLine(tree.DisplayTree());

            Console.WriteLine(); Console.WriteLine();
            Instance testInstance;
            Output output;
            Console.WriteLine("Testing Batman");
            testInstance = new Instance
            {
                Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("1", HERO_WEARS_MASK),
                            new Feature("1", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("1", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.WriteLine();

            Console.WriteLine("Testing Robin");
            testInstance = new Instance
            {
                Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("1", HERO_WEARS_MASK),
                            new Feature("1", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.WriteLine();

            Console.WriteLine("Testing Alfred");
            testInstance = new Instance
            {
                Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("0", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("1", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.WriteLine();

            Console.WriteLine("Testing Penguin");
            testInstance = new Instance
            {
                Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("0", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("1", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("1", HERO_SMOKES)
                        }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.WriteLine();

            Console.WriteLine("Testing Catwoman");
            testInstance = new Instance
            {
                Features =
                        {
                            new Feature("0", HERO_IS_MALE),
                            new Feature("1", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("1", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.WriteLine();

            Console.WriteLine("Testing Joker");
            testInstance = new Instance
            {
                Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("0", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.WriteLine();

            Console.WriteLine("Testing Batgirl");
            testInstance = new Instance
            {
                Features =
                {
                    new Feature("0", HERO_IS_MALE),
                    new Feature("1", HERO_WEARS_MASK),
                    new Feature("1", HERO_WEARS_CAPE),
                    new Feature("0", HERO_WEARS_TIE),
                    new Feature("1", HERO_WEARS_EARS),
                    new Feature("0", HERO_SMOKES)
                }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.WriteLine();

            Console.WriteLine("Testing Riddler");
            testInstance = new Instance
            {
                Features =
                {
                    new Feature("1", HERO_IS_MALE),
                    new Feature("1", HERO_WEARS_MASK),
                    new Feature("0", HERO_WEARS_CAPE),
                    new Feature("0", HERO_WEARS_TIE),
                    new Feature("0", HERO_WEARS_EARS),
                    new Feature("0", HERO_SMOKES)
                }
            };
            output = Tree.ProcessInstance(tree, testInstance);
            Console.WriteLine($"{output.Axis}: {output.Value}");

            Console.ReadLine();
        }

        const string FISH_CAN_SURVIVE_WITHOUT_SURFACING = "can survive without surfacing";
        const string FISH_HAS_FLIPPERS = "has flippers";
        const string FISH_IS_FISH = "is fish";

        const string HERO_IS_MALE = "is male";
        const string HERO_WEARS_MASK = "wears mask";
        const string HERO_WEARS_CAPE = "wears cape";
        const string HERO_WEARS_TIE = "wears tie";
        const string HERO_WEARS_EARS = "wears ears";
        const string HERO_SMOKES = "smokes";
        const string HERO_ALIGNMENT = "Alignment";

        static DecisionTreeSet GetHeroDataSet()
        {
            return new DecisionTreeSet
            {
                Instances =
                {
                    new Instance
                    {
                        Output = new Output("Good", HERO_ALIGNMENT),
                        Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("1", HERO_WEARS_MASK),
                            new Feature("1", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("1", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
                    },
                    new Instance
                    {
                        Output = new Output("Good", HERO_ALIGNMENT),
                        Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("1", HERO_WEARS_MASK),
                            new Feature("1", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
                    },
                    new Instance
                    {
                        Output = new Output("Good", HERO_ALIGNMENT),
                        Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("0", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("1", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
                    },
                    new Instance
                    {
                        Output = new Output("Bad", HERO_ALIGNMENT),
                        Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("0", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("1", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("1", HERO_SMOKES)
                        }
                    },
                    new Instance
                    {
                        Output = new Output("Bad", HERO_ALIGNMENT),
                        Features =
                        {
                            new Feature("0", HERO_IS_MALE),
                            new Feature("1", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("1", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
                    },
                    new Instance
                    {
                        Output = new Output("Bad", HERO_ALIGNMENT),
                        Features =
                        {
                            new Feature("1", HERO_IS_MALE),
                            new Feature("0", HERO_WEARS_MASK),
                            new Feature("0", HERO_WEARS_CAPE),
                            new Feature("0", HERO_WEARS_TIE),
                            new Feature("0", HERO_WEARS_EARS),
                            new Feature("0", HERO_SMOKES)
                        }
                    }
                }
            };
        }

        static DecisionTreeSet GetFishDataSet()
        {

            #region data

            var instance1 = new Instance
            {
                Output = new Output("yes", FISH_IS_FISH),
                Features = new List<Feature>
                                           {
                                               new Feature("1", FISH_CAN_SURVIVE_WITHOUT_SURFACING),
                                               new Feature("1", FISH_HAS_FLIPPERS)
                                           }
            };

            var instance2 = new Instance
            {
                Output = new Output("yes", FISH_IS_FISH),
                Features = new List<Feature>
                                           {
                                               new Feature("1", FISH_CAN_SURVIVE_WITHOUT_SURFACING),
                                               new Feature("1", FISH_HAS_FLIPPERS)
                                           }
            };

            var instance3 = new Instance
            {
                Output = new Output("no", FISH_IS_FISH),
                Features = new List<Feature>
                                           {
                                               new Feature("1", FISH_CAN_SURVIVE_WITHOUT_SURFACING),
                                               new Feature("0", FISH_HAS_FLIPPERS)
                                           }
            };

            var instance4 = new Instance
            {
                Output = new Output("no", FISH_IS_FISH),
                Features = new List<Feature>
                                           {
                                               new Feature("0", FISH_CAN_SURVIVE_WITHOUT_SURFACING),
                                               new Feature("1", FISH_HAS_FLIPPERS)
                                           }
            };

            var instance5 = new Instance
            {
                Output = new Output("no", FISH_IS_FISH),
                Features = new List<Feature>
                                           {
                                               new Feature("0", FISH_CAN_SURVIVE_WITHOUT_SURFACING),
                                               new Feature("1", FISH_HAS_FLIPPERS)
                                           }
            };

            #endregion

            return new DecisionTreeSet
            {
                Instances = new List<Instance>
                                          {
                                              instance1,
                                              instance2,
                                              instance3,
                                              instance4,
                                              instance5
                                          }
            };

        }
    }
}
