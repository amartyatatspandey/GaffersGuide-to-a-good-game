export interface MatchSetupPlayer {
  id: string;
  name: string;
  number: number;
  position: 'GK' | 'DF' | 'MF' | 'FW';
  isStarting: boolean;
}

export interface TeamSetup {
  name: string;
  formation: string;
  players: MatchSetupPlayer[];
}

export interface FixtureImportData {
  fixture_name: string;
  competition: string;
  team_a: TeamSetup;
  team_b: TeamSetup;
}

const REAL_LINEUPS: Record<string, FixtureImportData> = {
  'fix-1': {
    fixture_name: 'PSG vs Inter',
    competition: 'Champions League Final',
    team_a: {
      name: 'Paris Saint-Germain',
      formation: '4-3-3',
      players: [
        { id: 'psg-1', name: 'Gianluigi Donnarumma', number: 1, position: 'GK', isStarting: true },
        { id: 'psg-2', name: 'Achraf Hakimi', number: 2, position: 'DF', isStarting: true },
        { id: 'psg-3', name: 'Marquinhos', number: 5, position: 'DF', isStarting: true },
        { id: 'psg-4', name: 'Lucas Beraldo', number: 35, position: 'DF', isStarting: true },
        { id: 'psg-5', name: 'Nuno Mendes', number: 25, position: 'DF', isStarting: true },
        { id: 'psg-6', name: 'Warren Zaïre-Emery', number: 33, position: 'MF', isStarting: true },
        { id: 'psg-7', name: 'Vitinha', number: 17, position: 'MF', isStarting: true },
        { id: 'psg-8', name: 'Fabián Ruiz', number: 8, position: 'MF', isStarting: true },
        { id: 'psg-9', name: 'Ousmane Dembélé', number: 10, position: 'FW', isStarting: true },
        { id: 'psg-10', name: 'Bradley Barcola', number: 29, position: 'FW', isStarting: true },
        { id: 'psg-11', name: 'Randal Kolo Muani', number: 23, position: 'FW', isStarting: true },
        { id: 'psg-12', name: 'Matvey Safonov', number: 39, position: 'GK', isStarting: false },
        { id: 'psg-13', name: 'Milan Škriniar', number: 37, position: 'DF', isStarting: false },
        { id: 'psg-14', name: 'João Neves', number: 87, position: 'MF', isStarting: false },
        { id: 'psg-15', name: 'Marco Asensio', number: 11, position: 'FW', isStarting: false },
        { id: 'psg-16', name: 'Senny Mayulu', number: 41, position: 'MF', isStarting: false }
      ]
    },
    team_b: {
      name: 'Inter Milan',
      formation: '3-5-2',
      players: [
        { id: 'int-1', name: 'Yann Sommer', number: 1, position: 'GK', isStarting: true },
        { id: 'int-2', name: 'Benjamin Pavard', number: 28, position: 'DF', isStarting: true },
        { id: 'int-3', name: 'Francesco Acerbi', number: 15, position: 'DF', isStarting: true },
        { id: 'int-4', name: 'Alessandro Bastoni', number: 95, position: 'DF', isStarting: true },
        { id: 'int-5', name: 'Matteo Darmian', number: 36, position: 'DF', isStarting: true },
        { id: 'int-6', name: 'Nicolò Barella', number: 23, position: 'MF', isStarting: true },
        { id: 'int-7', name: 'Hakan Çalhanoğlu', number: 20, position: 'MF', isStarting: true },
        { id: 'int-8', name: 'Henrikh Mkhitaryan', number: 22, position: 'MF', isStarting: true },
        { id: 'int-9', name: 'Federico Dimarco', number: 32, position: 'DF', isStarting: true },
        { id: 'int-10', name: 'Marcus Thuram', number: 9, position: 'FW', isStarting: true },
        { id: 'int-11', name: 'Lautaro Martínez', number: 10, position: 'FW', isStarting: true },
        { id: 'int-12', name: 'Josep Martínez', number: 12, position: 'GK', isStarting: false },
        { id: 'int-13', name: 'Stefan de Vrij', number: 6, position: 'DF', isStarting: false },
        { id: 'int-14', name: 'Denzel Dumfries', number: 2, position: 'DF', isStarting: false },
        { id: 'int-15', name: 'Davide Frattesi', number: 16, position: 'MF', isStarting: false },
        { id: 'int-16', name: 'Mehdi Taremi', number: 99, position: 'FW', isStarting: false }
      ]
    }
  },
  'fix-2': {
    fixture_name: 'Real Madrid vs Man City',
    competition: 'Champions League Semi-Final',
    team_a: {
      name: 'Real Madrid',
      formation: '4-3-3',
      players: [
        { id: 'rm-1', name: 'Thibaut Courtois', number: 1, position: 'GK', isStarting: true },
        { id: 'rm-2', name: 'Dani Carvajal', number: 2, position: 'DF', isStarting: true },
        { id: 'rm-3', name: 'Éder Militão', number: 3, position: 'DF', isStarting: true },
        { id: 'rm-4', name: 'Antonio Rüdiger', number: 22, position: 'DF', isStarting: true },
        { id: 'rm-5', name: 'Ferland Mendy', number: 23, position: 'DF', isStarting: true },
        { id: 'rm-6', name: 'Federico Valverde', number: 8, position: 'MF', isStarting: true },
        { id: 'rm-7', name: 'Aurélien Tchouaméni', number: 14, position: 'MF', isStarting: true },
        { id: 'rm-8', name: 'Jude Bellingham', number: 5, position: 'MF', isStarting: true },
        { id: 'rm-9', name: 'Rodrygo', number: 11, position: 'FW', isStarting: true },
        { id: 'rm-10', name: 'Kylian Mbappé', number: 9, position: 'FW', isStarting: true },
        { id: 'rm-11', name: 'Vinícius Júnior', number: 7, position: 'FW', isStarting: true },
        { id: 'rm-12', name: 'Andriy Lunin', number: 13, position: 'GK', isStarting: false },
        { id: 'rm-13', name: 'Lucas Vázquez', number: 17, position: 'DF', isStarting: false },
        { id: 'rm-14', name: 'Luka Modrić', number: 10, position: 'MF', isStarting: false },
        { id: 'rm-15', name: 'Arda Güler', number: 15, position: 'MF', isStarting: false },
        { id: 'rm-16', name: 'Brahim Díaz', number: 21, position: 'FW', isStarting: false }
      ]
    },
    team_b: {
      name: 'Manchester City',
      formation: '4-3-3',
      players: [
        { id: 'mci-1', name: 'Ederson', number: 31, position: 'GK', isStarting: true },
        { id: 'mci-2', name: 'Kyle Walker', number: 2, position: 'DF', isStarting: true },
        { id: 'mci-3', name: 'Manuel Akanji', number: 25, position: 'DF', isStarting: true },
        { id: 'mci-4', name: 'Rúben Dias', number: 3, position: 'DF', isStarting: true },
        { id: 'mci-5', name: 'Josko Gvardiol', number: 24, position: 'DF', isStarting: true },
        { id: 'mci-6', name: 'Rodri', number: 16, position: 'MF', isStarting: true },
        { id: 'mci-7', name: 'Mateo Kovacic', number: 8, position: 'MF', isStarting: true },
        { id: 'mci-8', name: 'Kevin De Bruyne', number: 17, position: 'MF', isStarting: true },
        { id: 'mci-9', name: 'Bernardo Silva', number: 20, position: 'MF', isStarting: true },
        { id: 'mci-10', name: 'Phil Foden', number: 47, position: 'FW', isStarting: true },
        { id: 'mci-11', name: 'Erling Haaland', number: 9, position: 'FW', isStarting: true },
        { id: 'mci-12', name: 'Stefan Ortega', number: 18, position: 'GK', isStarting: false },
        { id: 'mci-13', name: 'John Stones', number: 5, position: 'DF', isStarting: false },
        { id: 'mci-14', name: 'Rico Lewis', number: 82, position: 'DF', isStarting: false },
        { id: 'mci-15', name: 'Ilkay Gündogan', number: 19, position: 'MF', isStarting: false },
        { id: 'mci-16', name: 'Jérémy Doku', number: 11, position: 'FW', isStarting: false }
      ]
    }
  },
  'fix-3': {
    fixture_name: 'Arsenal vs Chelsea',
    competition: 'Premier League',
    team_a: {
      name: 'Arsenal',
      formation: '4-3-3',
      players: [
        { id: 'ars-1', name: 'David Raya', number: 22, position: 'GK', isStarting: true },
        { id: 'ars-2', name: 'Ben White', number: 4, position: 'DF', isStarting: true },
        { id: 'ars-3', name: 'William Saliba', number: 2, position: 'DF', isStarting: true },
        { id: 'ars-4', name: 'Gabriel Magalhães', number: 6, position: 'DF', isStarting: true },
        { id: 'ars-5', name: 'Jurriën Timber', number: 12, position: 'DF', isStarting: true },
        { id: 'ars-6', name: 'Martin Ødegaard', number: 8, position: 'MF', isStarting: true },
        { id: 'ars-7', name: 'Thomas Partey', number: 5, position: 'MF', isStarting: true },
        { id: 'ars-8', name: 'Declan Rice', number: 41, position: 'MF', isStarting: true },
        { id: 'ars-9', name: 'Bukayo Saka', number: 7, position: 'FW', isStarting: true },
        { id: 'ars-10', name: 'Kai Havertz', number: 29, position: 'FW', isStarting: true },
        { id: 'ars-11', name: 'Gabriel Martinelli', number: 11, position: 'FW', isStarting: true },
        { id: 'ars-12', name: 'Neto', number: 32, position: 'GK', isStarting: false },
        { id: 'ars-13', name: 'Riccardo Calafiori', number: 33, position: 'DF', isStarting: false },
        { id: 'ars-14', name: 'Oleksandr Zinchenko', number: 17, position: 'DF', isStarting: false },
        { id: 'ars-15', name: 'Leandro Trossard', number: 19, position: 'FW', isStarting: false },
        { id: 'ars-16', name: 'Gabriel Jesus', number: 9, position: 'FW', isStarting: false }
      ]
    },
    team_b: {
      name: 'Chelsea',
      formation: '4-2-3-1',
      players: [
        { id: 'che-1', name: 'Robert Sánchez', number: 1, position: 'GK', isStarting: true },
        { id: 'che-2', name: 'Malo Gusto', number: 27, position: 'DF', isStarting: true },
        { id: 'che-3', name: 'Wesley Fofana', number: 29, position: 'DF', isStarting: true },
        { id: 'che-4', name: 'Levi Colwill', number: 6, position: 'DF', isStarting: true },
        { id: 'che-5', name: 'Marc Cucurella', number: 3, position: 'DF', isStarting: true },
        { id: 'che-6', name: 'Moisés Caicedo', number: 25, position: 'MF', isStarting: true },
        { id: 'che-7', name: 'Roméo Lavia', number: 45, position: 'MF', isStarting: true },
        { id: 'che-8', name: 'Noni Madueke', number: 11, position: 'FW', isStarting: true },
        { id: 'che-9', name: 'Cole Palmer', number: 20, position: 'MF', isStarting: true },
        { id: 'che-10', name: 'Pedro Neto', number: 7, position: 'FW', isStarting: true },
        { id: 'che-11', name: 'Nicolas Jackson', number: 15, position: 'FW', isStarting: true },
        { id: 'che-12', name: 'Filip Jørgensen', number: 12, position: 'GK', isStarting: false },
        { id: 'che-13', name: 'Tosin Adarabioyo', number: 4, position: 'DF', isStarting: false },
        { id: 'che-14', name: 'Renato Veiga', number: 40, position: 'DF', isStarting: false },
        { id: 'che-15', name: 'Enzo Fernández', number: 8, position: 'MF', isStarting: false },
        { id: 'che-16', name: 'Christopher Nkunku', number: 18, position: 'FW', isStarting: false }
      ]
    }
  },
  'fix-4': {
    fixture_name: 'Barcelona vs Real Madrid',
    competition: 'La Liga (El Clásico)',
    team_a: {
      name: 'Barcelona',
      formation: '4-3-3',
      players: [
        { id: 'bar-1', name: 'Marc-André ter Stegen', number: 1, position: 'GK', isStarting: true },
        { id: 'bar-2', name: 'Jules Koundé', number: 23, position: 'DF', isStarting: true },
        { id: 'bar-3', name: 'Pau Cubarsí', number: 2, position: 'DF', isStarting: true },
        { id: 'bar-4', name: 'Iñigo Martínez', number: 5, position: 'DF', isStarting: true },
        { id: 'bar-5', name: 'Alejandro Balde', number: 3, position: 'DF', isStarting: true },
        { id: 'bar-6', name: 'Pedri', number: 8, position: 'MF', isStarting: true },
        { id: 'bar-7', name: 'Marc Casadó', number: 17, position: 'MF', isStarting: true },
        { id: 'bar-8', name: 'Dani Olmo', number: 20, position: 'MF', isStarting: true },
        { id: 'bar-9', name: 'Lamine Yamal', number: 19, position: 'FW', isStarting: true },
        { id: 'bar-10', name: 'Robert Lewandowski', number: 9, position: 'FW', isStarting: true },
        { id: 'bar-11', name: 'Raphinha', number: 11, position: 'FW', isStarting: true },
        { id: 'bar-12', name: 'Iñaki Peña', number: 13, position: 'GK', isStarting: false },
        { id: 'bar-13', name: 'Frenkie de Jong', number: 21, position: 'MF', isStarting: false },
        { id: 'bar-14', name: 'Gavi', number: 6, position: 'MF', isStarting: false },
        { id: 'bar-15', name: 'Fermín López', number: 14, position: 'MF', isStarting: false },
        { id: 'bar-16', name: 'Ansu Fati', number: 10, position: 'FW', isStarting: false }
      ]
    },
    team_b: {
      name: 'Real Madrid',
      formation: '4-3-3',
      players: [
        { id: 'rmb-1', name: 'Thibaut Courtois', number: 1, position: 'GK', isStarting: true },
        { id: 'rmb-2', name: 'Dani Carvajal', number: 2, position: 'DF', isStarting: true },
        { id: 'rmb-3', name: 'Éder Militão', number: 3, position: 'DF', isStarting: true },
        { id: 'rmb-4', name: 'Antonio Rüdiger', number: 22, position: 'DF', isStarting: true },
        { id: 'rmb-5', name: 'Ferland Mendy', number: 23, position: 'DF', isStarting: true },
        { id: 'rmb-6', name: 'Federico Valverde', number: 8, position: 'MF', isStarting: true },
        { id: 'rmb-7', name: 'Aurélien Tchouaméni', number: 14, position: 'MF', isStarting: true },
        { id: 'rmb-8', name: 'Jude Bellingham', number: 5, position: 'MF', isStarting: true },
        { id: 'rmb-9', name: 'Rodrygo', number: 11, position: 'FW', isStarting: true },
        { id: 'rmb-10', name: 'Kylian Mbappé', number: 9, position: 'FW', isStarting: true },
        { id: 'rmb-11', name: 'Vinícius Júnior', number: 7, position: 'FW', isStarting: true },
        { id: 'rmb-12', name: 'Andriy Lunin', number: 13, position: 'GK', isStarting: false },
        { id: 'rmb-13', name: 'Luka Modrić', number: 10, position: 'MF', isStarting: false },
        { id: 'rmb-14', name: 'Arda Güler', number: 15, position: 'MF', isStarting: false },
        { id: 'rmb-15', name: 'Brahim Díaz', number: 21, position: 'FW', isStarting: false },
        { id: 'rmb-16', name: 'Endrick', number: 16, position: 'FW', isStarting: false }
      ]
    }
  },
  'fix-5': {
    fixture_name: 'Bayern Munich vs Dortmund',
    competition: 'Bundesliga (Der Klassiker)',
    team_a: {
      name: 'Bayern Munich',
      formation: '4-2-3-1',
      players: [
        { id: 'bay-1', name: 'Manuel Neuer', number: 1, position: 'GK', isStarting: true },
        { id: 'bay-2', name: 'Joshua Kimmich', number: 6, position: 'DF', isStarting: true },
        { id: 'bay-3', name: 'Dayot Upamecano', number: 2, position: 'DF', isStarting: true },
        { id: 'bay-4', name: 'Kim Min-jae', number: 3, position: 'DF', isStarting: true },
        { id: 'bay-5', name: 'Alphonso Davies', number: 19, position: 'DF', isStarting: true },
        { id: 'bay-6', name: 'Aleksandar Pavlović', number: 45, position: 'MF', isStarting: true },
        { id: 'bay-7', name: 'João Palhinha', number: 16, position: 'MF', isStarting: true },
        { id: 'bay-8', name: 'Michael Olise', number: 17, position: 'FW', isStarting: true },
        { id: 'bay-9', name: 'Jamal Musiala', number: 42, position: 'MF', isStarting: true },
        { id: 'bay-10', name: 'Serge Gnabry', number: 7, position: 'FW', isStarting: true },
        { id: 'bay-11', name: 'Harry Kane', number: 9, position: 'FW', isStarting: true },
        { id: 'bay-12', name: 'Sven Ulreich', number: 26, position: 'GK', isStarting: false },
        { id: 'bay-13', name: 'Eric Dier', number: 15, position: 'DF', isStarting: false },
        { id: 'bay-14', name: 'Raphaël Guerreiro', number: 22, position: 'DF', isStarting: false },
        { id: 'bay-15', name: 'Leroy Sané', number: 10, position: 'FW', isStarting: false },
        { id: 'bay-16', name: 'Thomas Müller', number: 25, position: 'MF', isStarting: false }
      ]
    },
    team_b: {
      name: 'Borussia Dortmund',
      formation: '4-3-3',
      players: [
        { id: 'dor-1', name: 'Gregor Kobel', number: 1, position: 'GK', isStarting: true },
        { id: 'dor-2', name: 'Julian Ryerson', number: 26, position: 'DF', isStarting: true },
        { id: 'dor-3', name: 'Waldemar Anton', number: 3, position: 'DF', isStarting: true },
        { id: 'dor-4', name: 'Nico Schlotterbeck', number: 4, position: 'DF', isStarting: true },
        { id: 'dor-5', name: 'Ramy Bensebaini', number: 5, position: 'DF', isStarting: true },
        { id: 'dor-6', name: 'Emre Can', number: 23, position: 'MF', isStarting: true },
        { id: 'dor-7', name: 'Pascal Groß', number: 10, position: 'MF', isStarting: true },
        { id: 'dor-8', name: 'Marcel Sabitzer', number: 20, position: 'MF', isStarting: true },
        { id: 'dor-9', name: 'Julian Brandt', number: 19, position: 'MF', isStarting: true },
        { id: 'dor-10', name: 'Jamie Gittens', number: 43, position: 'FW', isStarting: true },
        { id: 'dor-11', name: 'Serhou Guirassy', number: 9, position: 'FW', isStarting: true },
        { id: 'dor-12', name: 'Alexander Meyer', number: 33, position: 'GK', isStarting: false },
        { id: 'dor-13', name: 'Niklas Süle', number: 25, position: 'DF', isStarting: false },
        { id: 'dor-14', name: 'Felix Nmecha', number: 8, position: 'MF', isStarting: false },
        { id: 'dor-15', name: 'Maximilian Beier', number: 14, position: 'FW', isStarting: false },
        { id: 'dor-16', name: 'Donyell Malen', number: 21, position: 'FW', isStarting: false }
      ]
    }
  },
  'fix-6': {
    fixture_name: 'AC Milan vs Inter Milan',
    competition: 'Serie A (Derby della Madonnina)',
    team_a: {
      name: 'AC Milan',
      formation: '4-2-3-1',
      players: [
        { id: 'mil-1', name: 'Mike Maignan', number: 16, position: 'GK', isStarting: true },
        { id: 'mil-2', name: 'Emerson Royal', number: 22, position: 'DF', isStarting: true },
        { id: 'mil-3', name: 'Matteo Gabbia', number: 46, position: 'DF', isStarting: true },
        { id: 'mil-4', name: 'Fikayo Tomori', number: 23, position: 'DF', isStarting: true },
        { id: 'mil-5', name: 'Theo Hernández', number: 19, position: 'DF', isStarting: true },
        { id: 'mil-6', name: 'Youssouf Fofana', number: 29, position: 'MF', isStarting: true },
        { id: 'mil-7', name: 'Tijjani Reijnders', number: 14, position: 'MF', isStarting: true },
        { id: 'mil-8', name: 'Christian Pulisic', number: 11, position: 'FW', isStarting: true },
        { id: 'mil-9', name: 'Ruben Loftus-Cheek', number: 8, position: 'MF', isStarting: true },
        { id: 'mil-10', name: 'Rafael Leão', number: 10, position: 'FW', isStarting: true },
        { id: 'mil-11', name: 'Alvaro Morata', number: 7, position: 'FW', isStarting: true },
        { id: 'mil-12', name: 'Lorenzo Torriani', number: 96, position: 'GK', isStarting: false },
        { id: 'mil-13', name: 'Strahinja Pavlović', number: 31, position: 'DF', isStarting: false },
        { id: 'mil-14', name: 'Yunus Musah', number: 80, position: 'MF', isStarting: false },
        { id: 'mil-15', name: 'Samuel Chukwueze', number: 21, position: 'FW', isStarting: false },
        { id: 'mil-16', name: 'Tammy Abraham', number: 90, position: 'FW', isStarting: false }
      ]
    },
    team_b: {
      name: 'Inter Milan',
      formation: '3-5-2',
      players: [
        { id: 'int2-1', name: 'Yann Sommer', number: 1, position: 'GK', isStarting: true },
        { id: 'int2-2', name: 'Benjamin Pavard', number: 28, position: 'DF', isStarting: true },
        { id: 'int2-3', name: 'Francesco Acerbi', number: 15, position: 'DF', isStarting: true },
        { id: 'int2-4', name: 'Alessandro Bastoni', number: 95, position: 'DF', isStarting: true },
        { id: 'int2-5', name: 'Matteo Darmian', number: 36, position: 'DF', isStarting: true },
        { id: 'int2-6', name: 'Nicolò Barella', number: 23, position: 'MF', isStarting: true },
        { id: 'int2-7', name: 'Hakan Çalhanoğlu', number: 20, position: 'MF', isStarting: true },
        { id: 'int2-8', name: 'Henrikh Mkhitaryan', number: 22, position: 'MF', isStarting: true },
        { id: 'int2-9', name: 'Federico Dimarco', number: 32, position: 'DF', isStarting: true },
        { id: 'int2-10', name: 'Marcus Thuram', number: 9, position: 'FW', isStarting: true },
        { id: 'int2-11', name: 'Lautaro Martínez', number: 10, position: 'FW', isStarting: true },
        { id: 'int2-12', name: 'Josep Martínez', number: 12, position: 'GK', isStarting: false },
        { id: 'int2-13', name: 'Stefan de Vrij', number: 6, position: 'DF', isStarting: false },
        { id: 'int2-14', name: 'Denzel Dumfries', number: 2, position: 'DF', isStarting: false },
        { id: 'int2-15', name: 'Davide Frattesi', number: 16, position: 'MF', isStarting: false },
        { id: 'int2-16', name: 'Mehdi Taremi', number: 99, position: 'FW', isStarting: false }
      ]
    }
  }
};

/**
 * Fixture Import Service
 * Fetches real lineups for matches from our local repository.
 * Returns null if not found.
 */
export function getFixtureLineup(fixtureId: string): FixtureImportData | null {
  return REAL_LINEUPS[fixtureId] || null;
}
