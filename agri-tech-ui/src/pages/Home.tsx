import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import { Sprout, FlaskConical, TrendingUp, Linkedin, Github, Mail } from "lucide-react";
import heroImage from "@/assets/hero-agriculture.jpg";
import moduleCropImg from "@/assets/module-crop.jpg";
import moduleFertilizerImg from "@/assets/module-fertilizer.jpg";
import moduleYieldImg from "@/assets/module-yield.jpg";

const team = [
  {
    name: "Udhya Kumar K",
    linkedin: "https://www.linkedin.com/in/udhya-kumar-k-b7999128a/",
    github: "https://github.com/UdhyaKumarKMIT",
    email: "mailto:udhyak9445@gmail.com",
  },
  {
    name: "Mithun S",
    linkedin: "https://linkedin.com/in/member2",
    github: "https://github.com/member2",
    email: "mailto:member2@gmail.com",
  },
  {
    name: "Gopika ",
    
    linkedin: "www.linkedin.com/in/gopika-s-1523ab256",
    github: "https://github.com/gopikashreesakthi",
    email: "mailto:gopikashreesakthi@gmail.com",
  },
];

const Home = () => {
  const { t } = useTranslation();

  const features = [
    { to: "/crop-recommendation", label: t("hero.cropBtn"), icon: Sprout },
    { to: "/fertilizer-recommendation", label: t("hero.fertilizerBtn"), icon: FlaskConical },
    { to: "/yield-prediction", label: t("hero.yieldBtn"), icon: TrendingUp },
  ];

  const modules = [
    {
      icon: Sprout,
      title: t("modules.cropTitle"),
      description: t("modules.cropDesc"),
      image: moduleCropImg,
      to: "/crop-recommendation",
    },
    {
      icon: FlaskConical,
      title: t("modules.fertilizerTitle"),
      description: t("modules.fertilizerDesc"),
      image: moduleFertilizerImg,
      to: "/fertilizer-recommendation",
    },
    {
      icon: TrendingUp,
      title: t("modules.yieldTitle"),
      description: t("modules.yieldDesc"),
      image: moduleYieldImg,
      to: "/yield-prediction",
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative flex min-h-[70vh] items-center justify-center overflow-hidden">
        <img
          src={heroImage}
          alt="Agriculture landscape"
          className="absolute inset-0 h-full w-full object-cover"
        />
        <div className="hero-overlay absolute inset-0" />
        <div className="relative z-10 px-4 text-center">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="mb-4 font-display text-3xl font-extrabold text-primary-foreground sm:text-4xl md:text-5xl lg:text-6xl"
          >
            {t("hero.title")}
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 }}
            className="mb-8 text-lg text-primary-foreground/80"
          >
            {t("hero.subtitle")}
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="flex flex-wrap justify-center gap-3"
          >
            {features.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                className="flex items-center gap-2 rounded-lg bg-primary-foreground/95 px-5 py-3 text-sm font-semibold text-primary shadow-lg transition-all hover:scale-105 hover:bg-primary-foreground"
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            ))}
          </motion.div>
        </div>
      </section>

      {/* About Section */}
      <section className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mx-auto max-w-3xl text-center"
        >
          <h2 className="mb-4 font-display text-2xl font-bold text-foreground sm:text-3xl">
            {t("about.title")}
          </h2>
          <p className="text-base leading-relaxed text-muted-foreground sm:text-lg">
            {t("about.description")}
          </p>
        </motion.div>
      </section>

      {/* Modules Section */}
      <section className="bg-secondary/30 py-16">
        <div className="container mx-auto px-4">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center font-display text-2xl font-bold text-foreground sm:text-3xl"
          >
            {t("modules.sectionTitle")}
          </motion.h2>
          <div className="space-y-16">
            {modules.map((mod, i) => (
              <motion.div
                key={mod.to}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.1 }}
                className={`flex flex-col items-center gap-8 md:flex-row ${i % 2 === 1 ? "md:flex-row-reverse" : ""}`}
              >
                <div className="w-full md:w-1/2">
                  <img
                    src={mod.image}
                    alt={mod.title}
                    className="rounded-xl border border-border shadow-lg"
                  />
                </div>
                <div className="w-full md:w-1/2">
                  <div className="flex items-center gap-2 mb-3">
                    <mod.icon className="h-6 w-6 text-primary" />
                    <h3 className="font-display text-xl font-bold text-foreground sm:text-2xl">
                      {mod.title}
                    </h3>
                  </div>
                  <p className="mb-4 leading-relaxed text-muted-foreground">
                    {mod.description}
                  </p>
                  <Link
                    to={mod.to}
                    className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                  >
                    {t("modules.explore")}
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="container mx-auto px-4 py-16">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-12 text-center font-display text-2xl font-bold text-foreground sm:text-3xl"
        >
          {t("team.title")}
        </motion.h2>
        <div className="mx-auto grid max-w-4xl gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {team.map((member, i) => (
            <motion.div
              key={member.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="card-elevated flex flex-col items-center rounded-xl border border-border bg-card p-6 text-center"
            >
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
                <span className="text-2xl font-bold">{member.name.charAt(0)}</span>
              </div>
              <h3 className="text-lg font-bold text-foreground">{member.name}</h3>
              <div className="flex gap-3">
                <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="rounded-full bg-secondary p-2 text-foreground transition-colors hover:bg-primary hover:text-primary-foreground">
                  <Linkedin className="h-4 w-4" />
                </a>
                <a href={member.github} target="_blank" rel="noopener noreferrer" className="rounded-full bg-secondary p-2 text-foreground transition-colors hover:bg-primary hover:text-primary-foreground">
                  <Github className="h-4 w-4" />
                </a>
                <a href={member.email} className="rounded-full bg-secondary p-2 text-foreground transition-colors hover:bg-primary hover:text-primary-foreground">
                  <Mail className="h-4 w-4" />
                </a>
              </div>
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Home;
